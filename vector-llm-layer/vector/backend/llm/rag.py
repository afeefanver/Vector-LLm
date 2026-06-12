"""
llm/rag.py
----------
Retrieval-Augmented Generation layer.
Before every LLM call, we pull the most relevant chunks from ChromaDB
so the local model has accurate context — this is what pushes accuracy above 90%.

Flow:
  user query → embed → ChromaDB similarity search → top-K chunks
  → inject into prompt → LLM answers about YOUR data, not generic knowledge
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from llm.ollama_client import OllamaClient
from core.config import settings


class RAGRetriever:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.ollama  = OllamaClient()
        self.collection_name = settings.CHROMA_COLLECTION

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def ingest(self, text: str, doc_id: str, metadata: dict = None) -> None:
        """Chunk and store a document in ChromaDB."""
        chunks = self._chunk(text)
        collection = self._get_or_create_collection()

        for i, chunk in enumerate(chunks):
            embedding = await self.ollama.embed(chunk)
            collection.upsert(
                ids=[f"{doc_id}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{**(metadata or {}), "chunk_index": i}],
            )

    async def retrieve(self, query: str, n_results: int = None) -> str:
        """Return top-K relevant chunks as a single context string."""
        n = n_results or settings.MAX_CONTEXT_CHUNKS
        embedding = await self.ollama.embed(query)
        collection = self._get_or_create_collection()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(n, collection.count() or 1),
        )

        chunks = results["documents"][0] if results["documents"] else []
        if not chunks:
            return ""

        context = "\n\n---\n\n".join(chunks)
        return f"Relevant data context:\n{context}"

    @staticmethod
    def _chunk(text: str, size: int = 400, overlap: int = 80) -> list[str]:
        """Simple sliding-window chunker."""
        words  = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks


# Module-level singleton
retriever = RAGRetriever()
