"""
config.py
=========
All settings for the LLM microservice.
Change values here or override via .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama — local model server
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL:    str = "mistral"    # swap to phi3 or llama3.1 anytime
    OLLAMA_TIMEOUT:  int = 120          # seconds before timeout

    # ChromaDB — local vector store (used by rag.py, coming next)
    CHROMA_HOST:       str = "localhost"
    CHROMA_PORT:       int = 8000
    CHROMA_COLLECTION: str = "vector_docs"

    # Accuracy
    CONFIDENCE_THRESHOLD: float = 0.75  # below this → flag "needs more data"
    MAX_CONTEXT_CHUNKS:   int   = 6     # RAG chunks injected per prompt

    class Config:
        env_file = ".env"


settings = Settings()
