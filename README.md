# Qwen Model Serving Engine

This project serves as a lightweight LLM serving engine for Qwen models, using FastAPI and PyTorch. It supports continuous batching and paged attention for efficient and scalable inference.

## Features
- **FastAPI for HTTP serving**
- **Streaming and non-streaming responses**
- **Continuous batching**
- **Paged attention**

## Requirements
- Python 3.8+

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd qwen-serving-engine
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
```

## API Endpoints

### `POST /generate`
Generate text based on a given prompt.

- **Request**: `GenerationRequest`
- **Response**: `GenerationResponse`

### `POST /chat/completions`
OpenAI-compatible endpoint for chat completions.

- **Request**: `ChatCompletionRequest`
- **Response**: `ChatCompletionResponse`

## Example Request

Here's an example using `curl` to make a request to the `/generate` endpoint:

```bash
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?", "max_tokens": 50, "temperature": 0.7}'
```

## Testing

Run the tests using `pytest` to ensure everything works correctly:

```bash
pytest tests/
```
