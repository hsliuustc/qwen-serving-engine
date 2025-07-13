import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Mock the heavy model loading for tests
@pytest.fixture(scope="session")
def mock_model():
    """Mock model for testing without loading actual weights."""
    mock_model = Mock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.to.return_value = mock_model
    return mock_model

@pytest.fixture(scope="session")
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    mock_tokenizer.decode.return_value = "Mocked response text"
    return mock_tokenizer

@pytest.fixture(scope="session")
def test_client(mock_model, mock_tokenizer):
    """Create test client with mocked dependencies."""
    with patch('main.model', mock_model), \
         patch('main.tokenizer', mock_tokenizer):
        from main import app
        return TestClient(app)

@pytest.fixture
def sample_generation_request():
    """Sample generation request for testing."""
    return {
        "prompt": "Test prompt",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": False
    }

@pytest.fixture
def sample_chat_request():
    """Sample chat completion request for testing."""
    return {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
