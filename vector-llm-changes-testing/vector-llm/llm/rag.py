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

MULTI-TENANCY:
  Every user gets their own isolated ChromaDB collection:
      vector_docs_{user_id}
  This means User A's uploaded files are NEVER visible to User B.
  The user_id is extracted from the API key in auth.py and passed
  into every method here — nothing in this file reads from a shared
  collection anymore.

USAGE IN OTHER FILES:
    from llm.rag import rag

    context = await rag.retrieve("what was revenue in Q3?", user_id="u_abc123")
    await rag.ingest(text="...", doc_id="abc", user_id="u_abc123")

SETUP (run once):
  pip install chromadb
  chroma run --path ./chroma_data    ← in a separate terminal, port 8000
"""

import asyncio
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from llm.ollama import ollama_client
from config import settings


class RAGRetriever:

    def __init__(self):
        self._client      = None  # lazy-init so import never crashes
        self._collections: dict[str, object] = {}  # user_id → collection

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_client(self):
        """Connect to ChromaDB (lazy — only when first used)."""
        if self._client is None:
            self._client = chromadb.HttpClient(
                host     = settings.CHROMA_HOST,
                port     = settings.CHROMA_PORT,
                settings = ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self, user_id: str):
        """
        Get or create a per-user isolated collection.

        Collection name: vector_docs_{user_id}
        Each user's data is completely separate — no cross-user retrieval possible.
        Collections are cached in self._collections after first access.
        """
        if user_id not in self._collections:
            client = self._get_client()
            collection_name = f"vector_docs_{user_id}"
            self._collections[user_id] = client.get_or_create_collection(
                name     = collection_name,
                metadata = {"hnsw:space": "cosine"},
            )
        return self._collections[user_id]

    # ------------------------------------------------------------------
    # INGEST  — call this when a user uploads a file
    # ------------------------------------------------------------------

    async def ingest(
        self,
        text:     str,
        user_id:  str,
        doc_id:   str  = None,
        metadata: dict = None,
    ) -> dict:
        """
        Chunk a document and store it in the user's isolated ChromaDB collection.

        Args:
            text:     Full text content of the file (CSV, TXT, JSON, etc.)
            user_id:  The authenticated user's ID — determines which collection to write to.
            doc_id:   Unique ID for this document. Auto-generated if not provided.
            metadata: Extra info to store alongside chunks (filename, upload time, etc.)

        Returns:
            { "doc_id": str, "chunks_stored": int }

        Example:
            result = await rag.ingest(
                text="month,revenue\\nJan,50000\\nFeb,62000",
                user_id="u_abc123",
                doc_id="upload_001",
                metadata={"filename": "sales.csv"}
            )
        """
        doc_id     = doc_id or str(uuid.uuid4())
        metadata   = metadata or {}
        chunks     = self._chunk(text)
        collection = self._get_collection(user_id)

        for i, chunk in enumerate(chunks):
            chunk_id  = f"{doc_id}_chunk_{i}"
            embedding = await ollama_client.embed(chunk)

            # Wrap synchronous ChromaDB call so it doesn't block the event loop
            await asyncio.to_thread(
                collection.upsert,
                ids        = [chunk_id],
                embeddings = [embedding],
                documents  = [chunk],
                metadatas  = [{**metadata, "doc_id": doc_id, "chunk_index": i}],
            )

        print(f"[rag] user={user_id}  ingested doc_id={doc_id}  chunks={len(chunks)}")
        return {"doc_id": doc_id, "chunks_stored": len(chunks)}

    # ------------------------------------------------------------------
    # RETRIEVE  — call this before every LLM prompt
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, user_id: str, n_results: int = None) -> str:
        """
        Find the most relevant chunks for a query from the user's own collection only.

        Args:
            query:     The user's query (same text you'll send to the LLM)
            user_id:   The authenticated user's ID — only their data is searched.
            n_results: How many chunks to return. Defaults to MAX_CONTEXT_CHUNKS in config.

        Returns:
            A plain string ready to inject into a prompt.
            Returns "" if the user has not uploaded any documents yet.

        Example:
            context = await rag.retrieve("what was revenue in Q3?", user_id="u_abc123")
            prompt  = f"{context}\\n\\nUser question: what was revenue in Q3?"
        """
        collection = self._get_collection(user_id)

        # Wrap synchronous count() so it doesn't block the event loop
        total_docs = await asyncio.to_thread(collection.count)

        if total_docs == 0:
            return ""   # user hasn't uploaded anything yet

        n         = min(n_results or settings.MAX_CONTEXT_CHUNKS, total_docs)
        embedding = await ollama_client.embed(query)

        results = await asyncio.to_thread(
            collection.query,
            query_embeddings = [embedding],
            n_results        = n,
            include          = ["documents", "metadatas", "distances"],
        )

        chunks    = results["documents"][0]
        distances = results["distances"][0]

        # Filter out low-relevance chunks (cosine distance > 0.7 = not very similar)
        relevant = [
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

    async def delete_doc(self, doc_id: str, user_id: str) -> dict:
        """
        Remove all chunks for a specific document from the user's collection.
        Call this when a user deletes an uploaded file.
        """
        collection = self._get_collection(user_id)

        results = await asyncio.to_thread(
            collection.get,
            where={"doc_id": doc_id},
        )
        ids = results["ids"]

        if ids:
            await asyncio.to_thread(collection.delete, ids=ids)

        print(f"[rag] user={user_id}  deleted doc_id={doc_id}  chunks_removed={len(ids)}")
        return {"doc_id": doc_id, "chunks_removed": len(ids)}

    async def list_docs(self, user_id: str) -> list[dict]:
        """
        Return all documents ingested by this user.
        Surfaces doc_id + filename + chunk count so the frontend can show a file list.
        """
        collection = self._get_collection(user_id)
        results    = await asyncio.to_thread(
            collection.get,
            include=["metadatas"],
        )

        # De-duplicate by doc_id, pick the first metadata entry per doc
        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id":   doc_id,
                    "filename": meta.get("filename", "unknown"),
                }

        return list(seen.values())

    async def health_check(self, user_id: str = "_healthcheck") -> dict:
        """
        Check ChromaDB is reachable.
        Uses a throwaway collection name so no user data is affected.
        """
        try:
            collection = self._get_collection(user_id)
            count      = await asyncio.to_thread(collection.count)
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