from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import torch
from loguru import logger
import asyncio
from config import ServerConfig, GenerationRequest, ChatCompletionRequest, GenerationResponse, ChatCompletionResponse

app = FastAPI()

# Load configurations
config = ServerConfig.from_env()

# Load model and tokenizer
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.model.model_name,
    device_map="auto",  # Automatically decide device mapping
    torch_dtype=torch.float16
)
model.to(config.model.device)

async def generate_text(prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int, stream: bool):
    """
    Generate text based on the prompt, supports streaming.
    """
    logger.info("Generating response...")
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(config.model.device)
    # Continuous batching (not detailed)
    # Paged attention (not detailed)
    if not stream:
        outputs = model.generate(**inputs, max_length=max_tokens, top_k=top_k, top_p=top_p, temperature=temperature)
        text = tokenizer.decode(outputs[-1], skip_special_tokens=True)
        return GenerationResponse(text=text, tokens=len(outputs[-1]), finish_reason="stop")
    else:
        # Streaming implementation
        streamer = TextIteratorStreamer(tokenizer)
        model.generate(**inputs, max_length=max_tokens, top_k=top_k, top_p=top_p, temperature=temperature, streamer=streamer)
        for output in streamer:
            yield output

@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(request: GenerationRequest):
    """
    POST endpoint to generate text.
    """
    if request.stream:
        return StreamingResponse(generate_text(request.prompt, request.max_tokens, request.temperature, request.top_p, request.top_k, True),
                                 media_type="text/event-stream")
    else:
        return await generate_text(request.prompt, request.max_tokens, request.temperature, request.top_p, request.top_k, False)

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion_endpoint(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    """
    messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    if request.stream:
        return StreamingResponse(generate_text(messages_text, request.max_tokens, request.temperature, request.top_p, request.top_k, True),
                                 media_type="text/event-stream")
    else:
        return await generate_text(messages_text, request.max_tokens, request.temperature, request.top_p, request.top_k, False)
