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
    OLLAMA_MODEL:    str = "mistral"          # generative model — swap to phi3 or llama3.1 anytime
    EMBED_MODEL:     str = "nomic-embed-text" # dedicated embedding model — DO NOT use OLLAMA_MODEL here.
    #                                           nomic-embed-text is 3–5× faster than mistral for embeddings
    #                                           and produces much better vectors for RAG retrieval.
    #                                           Setup (run once): ollama pull nomic-embed-text
    OLLAMA_TIMEOUT:  int = 300          # seconds before timeout (CPU needs more time)

    # ChromaDB — local vector store
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    # Note: CHROMA_COLLECTION is intentionally absent.
    # Collections are per-user: vector_docs_{user_id}  (see llm/rag.py)

    # Accuracy
    CONFIDENCE_THRESHOLD: float = 0.75  # below this → flag "needs more data"
    MAX_CONTEXT_CHUNKS:   int   = 6     # RAG chunks injected per prompt

    class Config:
        env_file = ".env"


settings = Settings()
