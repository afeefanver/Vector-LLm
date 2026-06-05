"""
llm/rag.py
==========
Retrieval-Augmented Generation — the key to 90%+ accuracy.

WHY THIS EXISTS:
  A local 7B model doesn't know YOUR data.
  Before every LLM call, we pull the most relevant chunks from the user's
  uploaded files and inject them into the prompt as context.
  The model then answers about real data instead of guessing.

HOW IT WORKS:
  INGEST  (when user uploads a file)
    text → split into chunks → embed each chunk → store in ChromaDB

  RETRIEVE (before every LLM call)
    query → embed → ChromaDB similarity search → top-K chunks → context string

  USAGE IN OTHER FILES:
    from llm.rag import rag

    # Ingest a file
    await rag.ingest(text="...", doc_id="abc123", metadata={"filename": "sales.csv"})

    # Retrieve context before an LLM call
    context = await rag.retrieve("what was revenue in Q3?")
    # context is a plain string — inject it into your prompt

SETUP (run once):
  pip install chromadb
  chroma run --path ./chroma_data    ← in a separate terminal, port 8000
"""

import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from llm.ollama import ollama_client
from config import settings


class RAGRetriever:

    def __init__(self):
        self._client     = None   # lazy-init so import never crashes
        self._collection = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_client(self):
        """Connect to ChromaDB (lazy — only when first used)."""
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        """Get or create the collection (cosine similarity for text)."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # INGEST  — call this when a user uploads a file
    # ------------------------------------------------------------------

    async def ingest(
        self,
        text:     str,
        doc_id:   str = None,
        metadata: dict = None,
    ) -> dict:
        """
        Chunk a document and store it in ChromaDB.

        Args:
            text:     Full text content of the file (CSV, TXT, JSON, etc.)
            doc_id:   Unique ID for this document. Auto-generated if not provided.
            metadata: Extra info to store alongside chunks (filename, upload time, etc.)

        Returns:
            { "doc_id": str, "chunks_stored": int }

        Example:
            result = await rag.ingest(
                text="month,revenue\\nJan,50000\\nFeb,62000",
                doc_id="upload_001",
                metadata={"filename": "sales.csv"}
            )
        """
        doc_id     = doc_id or str(uuid.uuid4())
        metadata   = metadata or {}
        chunks     = self._chunk(text)
        collection = self._get_collection()

        for i, chunk in enumerate(chunks):
            chunk_id  = f"{doc_id}_chunk_{i}"
            embedding = await ollama_client.embed(chunk)

            collection.upsert(
                ids        = [chunk_id],
                embeddings = [embedding],
                documents  = [chunk],
                metadatas  = [{**metadata, "doc_id": doc_id, "chunk_index": i}],
            )

        print(f"[rag] ingested doc_id={doc_id}  chunks={len(chunks)}")
        return {"doc_id": doc_id, "chunks_stored": len(chunks)}

    # ------------------------------------------------------------------
    # RETRIEVE  — call this before every LLM prompt
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, n_results: int = None) -> str:
        """
        Find the most relevant chunks for a query and return them as a string.

        Args:
            query:     The user's query (same text you'll send to the LLM)
            n_results: How many chunks to return. Defaults to MAX_CONTEXT_CHUNKS in config.

        Returns:
            A plain string ready to inject into a prompt.
            Returns "" if no documents have been ingested yet.

        Example:
            context = await rag.retrieve("what was revenue in Q3?")
            prompt  = f"{context}\\n\\nUser question: what was revenue in Q3?"
        """
        collection = self._get_collection()
        total_docs = collection.count()

        if total_docs == 0:
            return ""   # nothing ingested yet — LLM will answer from general knowledge

        n          = min(n_results or settings.MAX_CONTEXT_CHUNKS, total_docs)
        embedding  = await ollama_client.embed(query)

        results    = collection.query(
            query_embeddings = [embedding],
            n_results        = n,
            include          = ["documents", "metadatas", "distances"],
        )

        chunks    = results["documents"][0]
        distances = results["distances"][0]

        # Filter out low-relevance chunks (cosine distance > 0.7 = not very similar)
        relevant  = [
            chunk for chunk, dist in zip(chunks, distances)
            if dist < 0.7
        ]

        if not relevant:
            return ""

        context = "\n\n---\n\n".join(relevant)
        return f"Relevant data from uploaded files:\n\n{context}"

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    async def delete_doc(self, doc_id: str) -> dict:
        """
        Remove all chunks for a specific document.
        Call this when a user deletes an uploaded file.
        """
        collection = self._get_collection()
        results    = collection.get(where={"doc_id": doc_id})
        ids        = results["ids"]

        if ids:
            collection.delete(ids=ids)

        print(f"[rag] deleted doc_id={doc_id}  chunks_removed={len(ids)}")
        return {"doc_id": doc_id, "chunks_removed": len(ids)}

    async def health_check(self) -> dict:
        """Check ChromaDB is reachable and return doc count."""
        try:
            collection = self._get_collection()
            count      = collection.count()
            return {"chroma_running": True, "chunks_stored": count}
        except Exception as e:
            return {
                "chroma_running": False,
                "error": str(e),
                "fix":   "Run: chroma run --path ./chroma_data",
            }

    # ------------------------------------------------------------------
    # INTERNAL — chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk(text: str, size: int = 400, overlap: int = 80) -> list[str]:
        """
        Split text into overlapping word-based chunks.

        size:    words per chunk (400 words ≈ 500 tokens, fits model context well)
        overlap: words shared between adjacent chunks (preserves context at boundaries)

        Example:
            chunk 1: words 0–399
            chunk 2: words 320–719   (80 word overlap with chunk 1)
            chunk 3: words 640–1039
        """
        words  = text.split()
        chunks = []
        i      = 0

        while i < len(words):
            chunk = " ".join(words[i : i + size])
            chunks.append(chunk)
            i += size - overlap

        return chunks


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.rag import rag
# ------------------------------------------------------------------
rag = RAGRetriever()
