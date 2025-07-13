from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loguru import logger
import asyncio
import uuid
import time
from typing import Dict, Any
from config import (
    ServerConfig, GenerationRequest, ChatCompletionRequest, 
    GenerationResponse, ChatCompletionResponse, ChatMessage
)
from batching import ContinuousBatchingEngine, GenerationRequest as BatchingRequest
from paged_attention import PagedKVCache, PagedAttention

app = FastAPI(
    title="Qwen Serving Engine",
    description="High-performance LLM serving engine with continuous batching and paged attention",
    version="1.0.0"
)

# Load configurations
config = ServerConfig.from_env()

# Global variables for model and engines
model = None
tokenizer = None
batching_engine = None
paged_kv_cache = None
paged_attention = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and engines on startup."""
    global model, tokenizer, batching_engine, paged_kv_cache, paged_attention
    
    logger.info("Starting Qwen Serving Engine...")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        device_map="auto",
        torch_dtype=getattr(torch, config.model.dtype)
    )
    model.to(config.model.device)
    
    # Initialize paged attention components
    # Note: These values should be extracted from the actual model config
    num_layers = len(model.transformer.h) if hasattr(model, 'transformer') else 32
    num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 32
    head_dim = model.config.hidden_size // num_heads if hasattr(model.config, 'hidden_size') else 128
    
    paged_kv_cache = PagedKVCache(
        config.paged_attention,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=config.model.device
    )
    
    paged_attention = PagedAttention(
        config.paged_attention,
        device=config.model.device
    )
    
    # Initialize continuous batching engine
    batching_engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=tokenizer,
        config=config.batching,
        device=config.model.device
    )
    
    # Start the batching engine
    await batching_engine.start()
    
    logger.info("Qwen Serving Engine started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global batching_engine
    
    if batching_engine:
        await batching_engine.stop()
    
    logger.info("Qwen Serving Engine stopped.")

async def process_generation_request(request: GenerationRequest) -> GenerationResponse:
    """Process a generation request using the batching engine."""
    global batching_engine
    
    # Convert to internal request format
    internal_request = BatchingRequest(
        request_id="",  # Will be set by batching engine
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop,
        stream=request.stream
    )
    
    # Add to batching engine
    request_id = await batching_engine.add_request(internal_request)
    
    # Wait for completion
    max_wait_time = 300  # 5 minutes timeout
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        result = await batching_engine.get_result(request_id)
        if result:
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return GenerationResponse(
                text=result["text"],
                tokens=result["tokens"],
                finish_reason=result["finish_reason"]
            )
        await asyncio.sleep(0.1)
    
    raise HTTPException(status_code=408, detail="Request timeout")

async def process_streaming_request(request: GenerationRequest):
    """Process a streaming generation request."""
    global batching_engine
    
    # Convert to internal request format
    internal_request = BatchingRequest(
        request_id="",
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop,
        stream=True
    )
    
    # Add to batching engine
    request_id = await batching_engine.add_request(internal_request)
    
    # Stream results
    async for chunk in batching_engine.stream_result(request_id):
        if chunk:
            yield f"data: {chunk}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(request: GenerationRequest):
    """
    POST endpoint to generate text using continuous batching.
    """
    if request.stream:
        return StreamingResponse(
            process_streaming_request(request),
            media_type="text/event-stream"
        )
    else:
        return await process_generation_request(request)

@app.post("/chat/completions")
async def chat_completion_endpoint(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    """
    # Convert chat messages to prompt format
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    
    prompt = "\n".join(prompt_parts) + "\nAssistant:"
    
    # Convert to generation request
    gen_request = GenerationRequest(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop
    )
    
    if request.stream:
        async def stream_chat_response():
            async for chunk in process_streaming_request(gen_request):
                # Format as OpenAI-compatible streaming response
                response_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.replace("data: ", "").replace("\n\n", "")},
                        "finish_reason": None
                    }]
                }
                if chunk.strip() == "data: [DONE]":
                    response_chunk["choices"][0]["finish_reason"] = "stop"
                    response_chunk["choices"][0]["delta"] = {}
                yield f"data: {response_chunk}\n\n"
        
        return StreamingResponse(stream_chat_response(), media_type="text/event-stream")
    else:
        result = await process_generation_request(gen_request)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text
                },
                "finish_reason": result.finish_reason
            }],
            usage={
                "prompt_tokens": 0,  # Would need to calculate
                "completion_tokens": result.tokens,
                "total_tokens": result.tokens
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global batching_engine
    
    if batching_engine and batching_engine.running:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "engine_stats": batching_engine.get_stats()
        }
    else:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "Batching engine not running"
        }

@app.get("/stats")
async def get_stats():
    """Get engine statistics."""
    global batching_engine, paged_attention
    
    stats = {
        "timestamp": time.time(),
        "config": {
            "model_name": config.model.model_name,
            "device": config.model.device,
            "max_batch_size": config.batching.max_batch_size,
            "block_size": config.paged_attention.block_size
        }
    }
    
    if batching_engine:
        stats["batching"] = batching_engine.get_stats()
    
    if paged_attention:
        stats["paged_attention"] = paged_attention.get_stats()
    
    return stats
