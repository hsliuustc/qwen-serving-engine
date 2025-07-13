"""
Continuous batching engine for efficient LLM inference.
Implements request queuing, batch formation, and processing.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import torch
from loguru import logger
from config import BatchingConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

class RequestStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationRequest:
    """Internal representation of a generation request."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    # Internal state
    status: RequestStatus = RequestStatus.PENDING
    created_at: float = field(default_factory=time.time)
    tokens_generated: int = 0
    input_tokens: Optional[torch.Tensor] = None
    generated_tokens: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    
    # Streaming
    response_queue: Optional[asyncio.Queue] = None

@dataclass
class BatchState:
    """State of a processing batch."""
    requests: List[GenerationRequest]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[Any] = None
    current_lengths: List[int] = field(default_factory=list)
    max_length: int = 0
    created_at: float = field(default_factory=time.time)

class ContinuousBatchingEngine:
    """
    Continuous batching engine for efficient LLM inference.
    
    Features:
    - Request queuing and batch formation
    - Dynamic batch size adjustment
    - Request prioritization
    - Memory-efficient processing
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: BatchingConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Request management
        self.pending_requests: asyncio.Queue = asyncio.Queue()
        self.processing_batches: List[BatchState] = []
        self.request_registry: Dict[str, GenerationRequest] = {}
        
        # Statistics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        
        # Engine state
        self.running = False
        self.engine_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized continuous batching engine with config: {config}")
    
    async def start(self):
        """Start the continuous batching engine."""
        if self.running:
            logger.warning("Engine is already running")
            return
        
        self.running = True
        self.engine_task = asyncio.create_task(self._engine_loop())
        logger.info("Continuous batching engine started")
    
    async def stop(self):
        """Stop the continuous batching engine."""
        if not self.running:
            return
        
        self.running = False
        if self.engine_task:
            self.engine_task.cancel()
            try:
                await self.engine_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Continuous batching engine stopped")
    
    async def add_request(self, request: GenerationRequest) -> str:
        """Add a new generation request to the queue."""
        self.total_requests += 1
        request.request_id = f"req_{self.total_requests}_{int(time.time() * 1000)}"
        
        if request.stream:
            request.response_queue = asyncio.Queue()
        
        self.request_registry[request.request_id] = request
        await self.pending_requests.put(request)
        
        logger.debug(f"Added request {request.request_id} to queue")
        return request.request_id
    
    async def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed request."""
        request = self.request_registry.get(request_id)
        if not request:
            return None
        
        if request.status == RequestStatus.COMPLETED:
            generated_text = self.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
            return {
                "text": generated_text,
                "tokens": len(request.generated_tokens),
                "finish_reason": request.finish_reason
            }
        elif request.status == RequestStatus.FAILED:
            return {
                "error": "Request failed during processing",
                "finish_reason": "error"
            }
        
        return None
    
    async def stream_result(self, request_id: str) -> AsyncGenerator[str, None]:
        """Stream the result of a request."""
        request = self.request_registry.get(request_id)
        if not request or not request.stream or not request.response_queue:
            return
        
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(request.response_queue.get(), timeout=1.0)
                    if chunk is None:  # End of stream marker
                        break
                    yield chunk
                except asyncio.TimeoutError:
                    if request.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                        break
                    continue
        except Exception as e:
            logger.error(f"Error streaming result for {request_id}: {e}")
    
    async def _engine_loop(self):
        """Main engine loop for continuous batching."""
        logger.info("Starting engine loop")
        
        while self.running:
            try:
                # Process existing batches
                await self._process_batches()
                
                # Form new batches from pending requests
                await self._form_new_batches()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in engine loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batches(self):
        """Process all active batches."""
        if not self.processing_batches:
            return
        
        completed_batches = []
        
        for batch in self.processing_batches:
            try:
                finished = await self._process_batch(batch)
                if finished:
                    completed_batches.append(batch)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Mark all requests in batch as failed
                for request in batch.requests:
                    request.status = RequestStatus.FAILED
                    request.finish_reason = "processing_error"
                completed_batches.append(batch)
        
        # Remove completed batches
        for batch in completed_batches:
            self.processing_batches.remove(batch)
    
    async def _process_batch(self, batch: BatchState) -> bool:
        """Process a single batch. Returns True if batch is complete."""
        active_requests = [req for req in batch.requests if not req.finished]
        
        if not active_requests:
            return True
        
        try:
            # Generate next tokens
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    max_length=batch.input_ids.shape[1] + 1,  # Generate one token at a time
                    past_key_values=batch.past_key_values,
                    do_sample=True,
                    temperature=active_requests[0].temperature,  # Use first request's params
                    top_p=active_requests[0].top_p,
                    top_k=active_requests[0].top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Process outputs for each request
            for i, request in enumerate(active_requests):
                if request.finished:
                    continue
                
                # Get new token
                new_token = outputs.sequences[i, -1].item()
                request.generated_tokens.append(new_token)
                request.tokens_generated += 1
                
                # Check stopping conditions
                should_stop = False
                finish_reason = None
                
                # Check max tokens
                if request.tokens_generated >= request.max_tokens:
                    should_stop = True
                    finish_reason = "length"
                
                # Check stop sequences
                if request.stop_sequences and not should_stop:
                    generated_text = self.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
                    for stop_seq in request.stop_sequences:
                        if stop_seq in generated_text:
                            should_stop = True
                            finish_reason = "stop"
                            break
                
                # Check EOS token
                if new_token == self.tokenizer.eos_token_id:
                    should_stop = True
                    finish_reason = "stop"
                
                # Handle streaming
                if request.stream and request.response_queue:
                    token_text = self.tokenizer.decode([new_token], skip_special_tokens=True)
                    await request.response_queue.put(token_text)
                
                # Finalize request if stopping
                if should_stop:
                    request.finished = True
                    request.status = RequestStatus.COMPLETED
                    request.finish_reason = finish_reason
                    self.completed_requests += 1
                    
                    if request.stream and request.response_queue:
                        await request.response_queue.put(None)  # End of stream marker
                    
                    logger.debug(f"Completed request {request.request_id} with {request.tokens_generated} tokens")
            
            # Update batch state
            batch.past_key_values = outputs.past_key_values
            batch.input_ids = outputs.sequences
            
            return all(req.finished for req in batch.requests)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Mark all requests as failed
            for request in active_requests:
                request.status = RequestStatus.FAILED
                request.finish_reason = "processing_error"
                self.failed_requests += 1
            return True
    
    async def _form_new_batches(self):
        """Form new batches from pending requests."""
        if self.pending_requests.empty():
            return
        
        # Collect requests for batching
        batch_requests = []
        total_tokens = 0
        
        # Try to get requests within timeout
        timeout = self.config.batch_timeout_ms / 1000.0
        start_time = time.time()
        
        while (
            len(batch_requests) < self.config.max_batch_size and
            total_tokens < self.config.max_batch_total_tokens and
            (time.time() - start_time) < timeout
        ):
            try:
                request = await asyncio.wait_for(
                    self.pending_requests.get(),
                    timeout=max(0.001, timeout - (time.time() - start_time))
                )
                
                # Tokenize request
                tokens = self.tokenizer.encode(request.prompt, return_tensors="pt")
                request.input_tokens = tokens
                
                # Check if we can fit this request
                request_tokens = tokens.shape[1]
                if total_tokens + request_tokens <= self.config.max_batch_total_tokens:
                    batch_requests.append(request)
                    total_tokens += request_tokens
                else:
                    # Put back the request if it doesn't fit
                    await self.pending_requests.put(request)
                    break
                    
            except asyncio.TimeoutError:
                break
        
        # Create batch if we have requests
        if batch_requests:
            await self._create_batch(batch_requests)
    
    async def _create_batch(self, requests: List[GenerationRequest]):
        """Create a new batch from requests."""
        # Prepare batch tensors
        input_ids_list = []
        attention_masks = []
        max_length = 0
        
        for request in requests:
            input_ids = request.input_tokens.squeeze(0)
            input_ids_list.append(input_ids)
            max_length = max(max_length, input_ids.shape[0])
        
        # Pad sequences to same length
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids in input_ids_list:
            pad_length = max_length - input_ids.shape[0]
            
            if pad_length > 0:
                padded_input = torch.cat([
                    torch.full((pad_length,), self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
                    input_ids
                ])
                attention_mask = torch.cat([torch.zeros(pad_length), torch.ones(input_ids.shape[0])])
            else:
                padded_input = input_ids
                attention_mask = torch.ones(input_ids.shape[0])
            
            padded_input_ids.append(padded_input)
            padded_attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batch_input_ids = torch.stack(padded_input_ids).to(self.device)
        batch_attention_mask = torch.stack(padded_attention_masks).to(self.device)
        
        # Create batch state
        batch = BatchState(
            requests=requests,
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            current_lengths=[len(req.input_tokens.squeeze(0)) for req in requests],
            max_length=max_length
        )
        
        # Mark requests as processing
        for request in requests:
            request.status = RequestStatus.PROCESSING
        
        self.processing_batches.append(batch)
        logger.debug(f"Created batch with {len(requests)} requests, total tokens: {batch_input_ids.numel()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "pending_requests": self.pending_requests.qsize(),
            "active_batches": len(self.processing_batches),
            "running": self.running
        }
