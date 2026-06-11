"""
llm/ollama.py
=============
The only file that talks to the local Ollama server.
No API key. No cost. Runs on your machine.

HOW IT WORKS:
  Ollama runs as a local HTTP server on port 11434.
  This client sends prompts to it and gets responses back.
  Your team never sees this — they only see your /query /dashboard /decide endpoints.

TWO MODELS ARE IN USE:
  OLLAMA_MODEL  — generative model (completions, streaming, decisions)
                  default: mistral  |  swap to phi3 or llama3.1 anytime
  EMBED_MODEL   — dedicated embedding model (RAG vector storage/retrieval)
                  default: nomic-embed-text  |  DO NOT use a generative model here.
                  Why: generative models are 3-5x slower for embeddings and produce
                  worse vectors. nomic-embed-text is purpose-built for this.

SETUP (run once in terminal before using):
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull mistral
  ollama pull nomic-embed-text    <- required for RAG

SWITCH GENERATIVE MODEL anytime -- just change OLLAMA_MODEL in .env:
  mistral       -> best for dev laptop (8-16 GB RAM, ~4 GB)
  phi3          -> lightest, fastest (~2 GB)
  llama3.1      -> best quality for production server

CONNECTION POOLING (H3 fix):
  A single httpx.AsyncClient is created at startup and reused for all requests.
  Previously a new client was created and destroyed on every call -- this caused
  a TCP handshake on every complete()/stream()/embed() call.
  The shared client keeps connections alive and eliminates that overhead.
  Call close() on app shutdown to drain the connection cleanly.
"""

import httpx
import json
from typing import AsyncGenerator
from config import settings


class OllamaClient:

    def __init__(self, timeout: int = None):
        self.base_url    = settings.OLLAMA_BASE_URL   # default: http://localhost:11434
        self.model       = settings.OLLAMA_MODEL       # generative model, default: mistral
        self.embed_model = settings.EMBED_MODEL        # H1: dedicated embedding model
        self.timeout     = timeout or settings.OLLAMA_TIMEOUT

        # H3: shared persistent client -- created once at startup, reused for every request.
        # Eliminates the TCP handshake cost that was previously paid on every API call.
        self._http = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        """Cleanly drain the shared HTTP client. Wire this to app shutdown in main.py."""
        await self._http.aclose()

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
            #                                       ^ low temp = consistent, accurate outputs
        }

        response = await self._http.post(
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

        async with self._http.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
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
        You won't call this directly -- rag.py calls it internally.

        H1 fix: uses self.embed_model (nomic-embed-text) instead of self.model (mistral).
        Mistral is a generative model -- it was never designed for embeddings.
        nomic-embed-text is a dedicated 137M embedding model: faster, better vectors.
        """
        response = await self._http.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},  # <- embed_model, not model
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def health_check(self) -> dict:
        """
        Check if Ollama is running and which models are available.
        Called by GET /health on your microservice.
        Also verifies both the generative model and embed model are pulled.
        """
        try:
            # health_check uses a short-lived client -- the shared self._http client
            # may not exist yet if called before __init__ completes (e.g. import-time tests).
            async with httpx.AsyncClient(timeout=3) as http:
                resp   = await http.get(f"{self.base_url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]

                def _is_pulled(name: str) -> bool:
                    return name in models or any(name in m for m in models)

                gen_ready   = _is_pulled(self.model)
                embed_ready = _is_pulled(self.embed_model)

                return {
                    "ollama_running":    True,
                    "active_model":      self.model,
                    "embed_model":       self.embed_model,
                    "model_ready":       gen_ready,
                    "embed_model_ready": embed_ready,
                    "available_models":  models,
                    # Surface a clear action if either model is missing
                    **({"fix": f"Run: ollama pull {self.embed_model}"} if not embed_ready else {}),
                }
        except Exception as e:
            return {
                "ollama_running": False,
                "error": str(e),
                "fix":   "Run: ollama serve   (in a separate terminal)",
            }


# -----------------------------------------------------------------------
# Module-level singleton -- import this everywhere else
# Usage:  from llm.ollama import ollama_client
# -----------------------------------------------------------------------
ollama_client = OllamaClient()
