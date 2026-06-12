"""
llm/ollama_client.py
--------------------
Thin async wrapper around Ollama's REST API.
Runs 100% locally — no API key, no per-token cost.

Swap OLLAMA_MODEL in .env to change the model:
  Dev (8-16GB RAM):   mistral  or  phi3
  Production server:  llama3.1  or  mixtral
"""

import httpx
import json
from typing import AsyncGenerator
from core.config import settings


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model    = settings.OLLAMA_MODEL
        self.timeout  = settings.OLLAMA_TIMEOUT

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Single-shot completion. Returns full response as string."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streaming completion — yields tokens as they arrive.
        Use this for chat-style responses in the frontend."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for RAG / ChromaDB ingestion."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def is_available(self) -> bool:
        """Health check — returns True if Ollama is running locally."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
