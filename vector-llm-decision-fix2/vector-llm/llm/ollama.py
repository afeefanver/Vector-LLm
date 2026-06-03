"""
llm/ollama.py
=============
The only file that talks to the local Mistral model.
No API key. No cost. Runs on your machine via Ollama.

HOW IT WORKS:
  Ollama runs as a local HTTP server on port 11434.
  This client sends prompts to it and gets responses back.
  Your team never sees this — they only see your /query /dashboard /decide endpoints.

SETUP (run once in terminal before using):
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull mistral

SWITCH MODEL anytime — just change OLLAMA_MODEL in .env:
  mistral       → best for dev laptop (8–16 GB RAM, ~4 GB)
  phi3          → lightest, fastest (~2 GB)
  llama3.1      → best quality for production server
"""

import httpx
import json
from typing import AsyncGenerator
from config import settings


class OllamaClient:

    def __init__(self, timeout: int = None):
        self.base_url = settings.OLLAMA_BASE_URL   # default: http://localhost:11434
        self.model    = settings.OLLAMA_MODEL       # default: mistral
        self.timeout  = timeout or settings.OLLAMA_TIMEOUT

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Send a prompt, get the full response back as a string.
        Use this for: intent classification, dashboard specs, decisions.

        Example:
            client = OllamaClient()
            answer = await client.complete("Summarise this data: ...")
        """
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
            #                                       ↑ low temp = consistent, accurate outputs
        }

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            response = await http.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["response"]

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Send a prompt and get tokens back one by one as they're generated.
        Use this for: the /query endpoint when the frontend wants live streaming.

        Example:
            async for token in client.stream("Explain this chart..."):
                print(token, end="", flush=True)
        """
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  True,
            "options": {"temperature": 0.3},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            async with http.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    async def embed(self, text: str) -> list[float]:
        """
        Convert text into a vector (list of floats).
        Used by rag.py to store and search documents in ChromaDB.
        You won't call this directly — rag.py calls it internally.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            response = await http.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def health_check(self) -> dict:
        """
        Check if Ollama is running and which models are available.
        Called by GET /health on your microservice.
        """
        try:
            async with httpx.AsyncClient(timeout=3) as http:
                resp = await http.get(f"{self.base_url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return {
                    "ollama_running": True,
                    "active_model":   self.model,
                    "available_models": models,
                    "model_ready": self.model in models or any(self.model in m for m in models),
                }
        except Exception as e:
            return {
                "ollama_running": False,
                "error": str(e),
                "fix":   "Run: ollama serve   (in a separate terminal)",
            }


# -----------------------------------------------------------------------
# Module-level singleton — import this everywhere else
# Usage:  from llm.ollama import ollama_client
# -----------------------------------------------------------------------
ollama_client = OllamaClient()
