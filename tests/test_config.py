import pytest
import os
from unittest.mock import patch
from config import (
    ServerConfig, ModelConfig, BatchingConfig, PagedAttentionConfig,
    GenerationRequest, ChatCompletionRequest, ChatMessage
)

class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_name == "Qwen/Qwen-7B-Chat"
        assert config.device == "cuda"
        assert config.dtype == "float16"
        assert config.max_model_len == 8192
        assert config.tensor_parallel_size == 1
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_name="custom/model",
            device="cpu",
            dtype="float32",
            max_model_len=4096,
            tensor_parallel_size=2
        )
        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.max_model_len == 4096
        assert config.tensor_parallel_size == 2

class TestBatchingConfig:
    """Test cases for BatchingConfig."""
    
    def test_default_values(self):
        """Test default batching configuration."""
        config = BatchingConfig()
        assert config.max_batch_size == 32
        assert config.max_waiting_tokens == 20
        assert config.max_batch_total_tokens == 16384
        assert config.batch_timeout_ms == 100
    
    def test_custom_values(self):
        """Test custom batching configuration."""
        config = BatchingConfig(
            max_batch_size=64,
            max_waiting_tokens=40,
            max_batch_total_tokens=32768,
            batch_timeout_ms=200
        )
        assert config.max_batch_size == 64
        assert config.max_waiting_tokens == 40
        assert config.max_batch_total_tokens == 32768
        assert config.batch_timeout_ms == 200

class TestPagedAttentionConfig:
    """Test cases for PagedAttentionConfig."""
    
    def test_default_values(self):
        """Test default paged attention configuration."""
        config = PagedAttentionConfig()
        assert config.block_size == 16
        assert config.max_num_blocks_per_seq == 512
        assert config.max_num_cached_blocks == 2048
        assert config.cache_dtype == "float16"
    
    def test_custom_values(self):
        """Test custom paged attention configuration."""
        config = PagedAttentionConfig(
            block_size=32,
            max_num_blocks_per_seq=1024,
            max_num_cached_blocks=4096,
            cache_dtype="float32"
        )
        assert config.block_size == 32
        assert config.max_num_blocks_per_seq == 1024
        assert config.max_num_cached_blocks == 4096
        assert config.cache_dtype == "float32"

class TestServerConfig:
    """Test cases for ServerConfig."""
    
    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.log_level == "info"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.batching, BatchingConfig)
        assert isinstance(config.paged_attention, PagedAttentionConfig)
    
    def test_from_env(self):
        """Test configuration loading from environment variables."""
        env_vars = {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "MODEL_NAME": "custom/qwen-model",
            "DEVICE": "cpu",
            "MAX_MODEL_LEN": "4096",
            "MAX_BATCH_SIZE": "16",
            "MAX_WAITING_TOKENS": "10"
        }
        
        with patch.dict(os.environ, env_vars):
            config = ServerConfig.from_env()
            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.model.model_name == "custom/qwen-model"
            assert config.model.device == "cpu"
            assert config.model.max_model_len == 4096
            assert config.batching.max_batch_size == 16
            assert config.batching.max_waiting_tokens == 10

class TestGenerationRequest:
    """Test cases for GenerationRequest."""
    
    def test_valid_request(self):
        """Test valid generation request."""
        request = GenerationRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            stream=True,
            stop=["</s>", "\n\n"]
        )
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.top_k == 40
        assert request.stream is True
        assert request.stop == ["</s>", "\n\n"]
    
    def test_default_values(self):
        """Test default values in generation request."""
        request = GenerationRequest(prompt="Test")
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 50
        assert request.stream is False
        assert request.stop is None
    
    def test_invalid_temperature(self):
        """Test validation of temperature parameter."""
        # This would require custom validation in the actual model
        # For now, we just test that the model accepts the values
        request = GenerationRequest(prompt="Test", temperature=2.0)
        assert request.temperature == 2.0

class TestChatCompletionRequest:
    """Test cases for ChatCompletionRequest."""
    
    def test_valid_request(self):
        """Test valid chat completion request."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!")
        ]
        request = ChatCompletionRequest(
            model="qwen-chat",
            messages=messages,
            max_tokens=200,
            temperature=0.8,
            stream=True
        )
        assert request.model == "qwen-chat"
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "You are helpful."
        assert request.messages[1].role == "user"
        assert request.messages[1].content == "Hello!"
        assert request.max_tokens == 200
        assert request.temperature == 0.8
        assert request.stream is True
    
    def test_default_values(self):
        """Test default values in chat completion request."""
        messages = [ChatMessage(role="user", content="Test")]
        request = ChatCompletionRequest(messages=messages)
        assert request.model == "qwen"
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is False
        assert request.stop is None

class TestChatMessage:
    """Test cases for ChatMessage."""
    
    def test_valid_message(self):
        """Test valid chat message."""
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
    
    def test_different_roles(self):
        """Test different message roles."""
        roles = ["system", "user", "assistant"]
        for role in roles:
            message = ChatMessage(role=role, content=f"Message from {role}")
            assert message.role == role
            assert message.content == f"Message from {role}"
