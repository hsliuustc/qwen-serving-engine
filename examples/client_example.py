#!/usr/bin/env python3
"""
Example client for the Qwen Model Serving Engine.
Demonstrates both sync and async usage patterns.
"""

import asyncio
import json
import requests
import aiohttp
from typing import Dict, Any

# Server configuration
SERVER_URL = "http://localhost:8000"

def sync_generate_example():
    """Example of synchronous text generation."""
    print("=== Synchronous Generation Example ===")
    
    # Basic generation request
    request_data = {
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/generate", json=request_data)
        response.raise_for_status()
        result = response.json()
        
        print(f"Generated text: {result['text']}")
        print(f"Tokens: {result['tokens']}")
        print(f"Finish reason: {result['finish_reason']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

def sync_chat_completion_example():
    """Example of synchronous chat completion."""
    print("\n=== Synchronous Chat Completion Example ===")
    
    # Chat completion request
    request_data = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke about programming."}
        ],
        "max_tokens": 150,
        "temperature": 0.8,
        "stream": False
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/chat/completions", json=request_data)
        response.raise_for_status()
        result = response.json()
        
        print(f"Response: {json.dumps(result, indent=2)}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

def sync_streaming_example():
    """Example of synchronous streaming generation."""
    print("\n=== Synchronous Streaming Example ===")
    
    request_data = {
        "prompt": "Write a short story about a robot:",
        "max_tokens": 200,
        "temperature": 0.8,
        "stream": True
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/generate", json=request_data, stream=True)
        response.raise_for_status()
        
        print("Streaming response:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print()  # New line after streaming
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

async def async_generate_example():
    """Example of asynchronous text generation."""
    print("\n=== Asynchronous Generation Example ===")
    
    request_data = {
        "prompt": "Explain quantum computing in simple terms:",
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{SERVER_URL}/generate", json=request_data) as response:
                response.raise_for_status()
                result = await response.json()
                
                print(f"Generated text: {result['text']}")
                print(f"Tokens: {result['tokens']}")
                
        except aiohttp.ClientError as e:
            print(f"Error: {e}")

async def async_streaming_example():
    """Example of asynchronous streaming generation."""
    print("\n=== Asynchronous Streaming Example ===")
    
    request_data = {
        "prompt": "List the benefits of renewable energy:",
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{SERVER_URL}/generate", json=request_data) as response:
                response.raise_for_status()
                
                print("Streaming response:")
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        print(chunk.decode('utf-8'), end='', flush=True)
                print()  # New line after streaming
                
        except aiohttp.ClientError as e:
            print(f"Error: {e}")

def batch_requests_example():
    """Example of sending multiple requests concurrently."""
    print("\n=== Batch Requests Example ===")
    
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What are the applications of AI?",
        "How does deep learning work?",
        "What is natural language processing?"
    ]
    
    async def make_request(session: aiohttp.ClientSession, prompt: str) -> Dict[str, Any]:
        request_data = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            async with session.post(f"{SERVER_URL}/generate", json=request_data) as response:
                response.raise_for_status()
                result = await response.json()
                return {"prompt": prompt, "response": result['text'][:100] + "..."}
        except aiohttp.ClientError as e:
            return {"prompt": prompt, "error": str(e)}
    
    async def run_batch():
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Prompt: {result['prompt']}")
                if 'response' in result:
                    print(f"   Response: {result['response']}")
                else:
                    print(f"   Error: {result['error']}")
                print()
    
    asyncio.run(run_batch())

def health_check_example():
    """Example of health check endpoint."""
    print("\n=== Health Check Example ===")
    
    try:
        response = requests.get(f"{SERVER_URL}/docs")
        if response.status_code == 200:
            print("✓ Server is running and healthy")
            print(f"  API documentation available at: {SERVER_URL}/docs")
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Server is not accessible: {e}")

if __name__ == "__main__":
    print("Qwen Model Serving Engine - Client Examples")
    print("=" * 50)
    
    # Check server health first
    health_check_example()
    
    # Run synchronous examples
    sync_generate_example()
    sync_chat_completion_example()
    sync_streaming_example()
    
    # Run asynchronous examples
    asyncio.run(async_generate_example())
    asyncio.run(async_streaming_example())
    
    # Run batch requests example
    batch_requests_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
