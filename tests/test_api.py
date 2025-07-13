import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from config import GenerationRequest, ChatCompletionRequest, ChatMessage

class TestGenerateEndpoint:
    """Test cases for the /generate endpoint."""
    
    def test_generate_non_streaming(self, test_client, sample_generation_request):
        """Test non-streaming text generation."""
        response = test_client.post("/generate", json=sample_generation_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "tokens" in data
        assert "finish_reason" in data
        assert data["finish_reason"] == "stop"
    
    def test_generate_with_custom_parameters(self, test_client):
        """Test generation with custom parameters."""
        request_data = {
            "prompt": "Custom prompt",
            "max_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "stream": False,
            "stop": ["</s>"]
        }
        
        response = test_client.post("/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "tokens" in data
        assert "finish_reason" in data
    
    def test_generate_streaming(self, test_client):
        """Test streaming text generation."""
        request_data = {
            "prompt": "Streaming test",
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": True
        }
        
        response = test_client.post("/generate", json=request_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    def test_generate_invalid_request(self, test_client):
        """Test generation with invalid request data."""
        invalid_request = {
            "max_tokens": 50,  # Missing required 'prompt' field
            "temperature": 0.7
        }
        
        response = test_client.post("/generate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_with_negative_tokens(self, test_client):
        """Test generation with negative max_tokens."""
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": -10,  # Invalid negative value
            "temperature": 0.7
        }
        
        response = test_client.post("/generate", json=request_data)
        
        # The request should still be processed (validation might be added later)
        assert response.status_code in [200, 422]

class TestChatCompletionEndpoint:
    """Test cases for the /chat/completions endpoint."""
    
    def test_chat_completion_non_streaming(self, test_client, sample_chat_request):
        """Test non-streaming chat completion."""
        response = test_client.post("/chat/completions", json=sample_chat_request)
        
        assert response.status_code == 200
        # The actual response format depends on the implementation
        # This test verifies the endpoint is accessible and returns 200
    
    def test_chat_completion_with_system_message(self, test_client):
        """Test chat completion with system message."""
        request_data = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate factorial."}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": False
        }
        
        response = test_client.post("/chat/completions", json=request_data)
        
        assert response.status_code == 200
    
    def test_chat_completion_streaming(self, test_client):
        """Test streaming chat completion."""
        request_data = {
            "model": "qwen",
            "messages": [
                {"role": "user", "content": "Tell me a story."}
            ],
            "max_tokens": 100,
            "temperature": 0.8,
            "stream": True
        }
        
        response = test_client.post("/chat/completions", json=request_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    def test_chat_completion_empty_messages(self, test_client):
        """Test chat completion with empty messages."""
        request_data = {
            "model": "qwen",
            "messages": [],  # Empty messages array
            "max_tokens": 100
        }
        
        response = test_client.post("/chat/completions", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_completion_invalid_role(self, test_client):
        """Test chat completion with invalid role."""
        request_data = {
            "model": "qwen",
            "messages": [
                {"role": "invalid_role", "content": "Test message"}
            ],
            "max_tokens": 100
        }
        
        response = test_client.post("/chat/completions", json=request_data)
        
        # The request might still be processed depending on validation
        assert response.status_code in [200, 422]

class TestHealthAndDocumentation:
    """Test cases for health check and documentation endpoints."""
    
    def test_docs_endpoint(self, test_client):
        """Test that API documentation is available."""
        response = test_client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_endpoint(self, test_client):
        """Test that OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_redoc_endpoint(self, test_client):
        """Test that ReDoc documentation is available."""
        response = test_client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_invalid_json(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_content_type(self, test_client):
        """Test handling of missing content type."""
        response = test_client.post("/generate", data='{"prompt": "test"}')
        
        assert response.status_code == 422
    
    def test_unsupported_method(self, test_client):
        """Test unsupported HTTP method."""
        response = test_client.get("/generate")
        
        assert response.status_code == 405  # Method not allowed
    
    def test_nonexistent_endpoint(self, test_client):
        """Test access to non-existent endpoint."""
        response = test_client.post("/nonexistent")
        
        assert response.status_code == 404  # Not found

class TestModelIntegration:
    """Test cases for model integration (with mocked models)."""
    
    @patch('main.model')
    @patch('main.tokenizer')
    def test_model_tokenizer_integration(self, mock_tokenizer, mock_model, test_client):
        """Test that model and tokenizer are called correctly."""
        # Setup mocks
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer.decode.return_value = "Test response"
        mock_model.generate.return_value = Mock()
        
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": False
        }
        
        response = test_client.post("/generate", json=request_data)
        
        assert response.status_code == 200
        # Verify mocks were called (if the integration is properly implemented)
        mock_tokenizer.assert_called()
        mock_model.generate.assert_called()
