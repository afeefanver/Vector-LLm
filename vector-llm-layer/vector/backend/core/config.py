from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Vector"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/vector"

    # Ollama — local LLM, zero API cost
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"          # change to llama3.1 on production server
    OLLAMA_TIMEOUT: int = 120

    # ChromaDB — local vector store for RAG grounding
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "vector_docs"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Accuracy
    CONFIDENCE_THRESHOLD: float = 0.75    # below this → ask for more data
    MAX_CONTEXT_CHUNKS: int = 6           # RAG chunks injected into prompt

    class Config:
        env_file = ".env"

settings = Settings()
