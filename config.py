from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import os

class ModelConfig(BaseModel):
    """Configuration for the Qwen model"""
    model_name: str = Field(default="Qwen/Qwen-7B-Chat", description="Model name or path")
    device: str = Field(default="cuda", description="Device to load model on")
    dtype: str = Field(default="float16", description="Data type for model weights")
    max_model_len: int = Field(default=8192, description="Maximum model sequence length")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    
class BatchingConfig(BaseModel):
    """Configuration for continuous batching"""
    max_batch_size: int = Field(default=32, description="Maximum batch size for inference")
    max_waiting_tokens: int = Field(default=20, description="Maximum tokens to wait before processing batch")
    max_batch_total_tokens: int = Field(default=16384, description="Maximum total tokens in a batch")
    batch_timeout_ms: int = Field(default=100, description="Timeout for batch formation in milliseconds")
    
class PagedAttentionConfig(BaseModel):
    """Configuration for paged attention"""
    block_size: int = Field(default=16, description="Block size for paged attention")
    max_num_blocks_per_seq: int = Field(default=512, description="Maximum blocks per sequence")
    max_num_cached_blocks: int = Field(default=2048, description="Maximum cached blocks")
    cache_dtype: str = Field(default="float16", description="Data type for KV cache")
    
class ServerConfig(BaseModel):
    """Main server configuration"""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    log_level: str = Field(default="info", description="Log level")
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    paged_attention: PagedAttentionConfig = Field(default_factory=PagedAttentionConfig)
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            model=ModelConfig(
                model_name=os.getenv("MODEL_NAME", "Qwen/Qwen-7B-Chat"),
                device=os.getenv("DEVICE", "cuda"),
                max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
            ),
            batching=BatchingConfig(
                max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "32")),
                max_waiting_tokens=int(os.getenv("MAX_WAITING_TOKENS", "20")),
            ),
        )

# Request/Response Models
class GenerationRequest(BaseModel):
    """Request for text generation"""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    stream: bool = Field(default=False, description="Whether to stream response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    
class GenerationResponse(BaseModel):
    """Response from text generation"""
    text: str = Field(..., description="Generated text")
    tokens: int = Field(..., description="Number of tokens generated")
    finish_reason: str = Field(..., description="Reason for completion")
    
class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    
class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(default="qwen", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    stream: bool = Field(default=False, description="Whether to stream response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    
class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
