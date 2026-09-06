"""
tests/test_rag.py
Tests for RAGRetriever — per-user collection isolation (C1), async wrapping (H4).

Key signatures confirmed from source:
  ingest(text, user_id, doc_id=None, metadata=None)
  retrieve(query, user_id, n_results=None)
  delete_doc(doc_id, user_id)
  list_docs(user_id)
  _get_collection(user_id) — calls _get_client() internally (lazy init)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

USER_A   = "u_001"
USER_B   = "u_002"
TEST_USER = "u_test"

SAMPLE_TEXT = "month,revenue\nJan,50000\nFeb,62000\nMar,71000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chroma_collection():
    col = MagicMock()
    col.count   = MagicMock(return_value=3)
    col.upsert  = MagicMock()
    col.delete  = MagicMock()
    col.query   = MagicMock(return_value={
        "documents": [["Revenue Q1 ₹12 lakh"]],
        "ids":       [["doc_1_chunk_0"]],
        "metadatas": [[{"doc_id": "doc_1", "filename": "report.csv", "chunk_index": 0}]],
        "distances": [[0.12]],
    })
    col.get = MagicMock(return_value={
        "ids":       ["doc_1_chunk_0", "doc_2_chunk_0"],
        "metadatas": [
            {"doc_id": "doc_1", "filename": "report.csv", "chunk_index": 0},
            {"doc_id": "doc_2", "filename": "sales.csv",  "chunk_index": 0},
        ],
    })
    return col


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    client = MagicMock()
    client.get_or_create_collection = MagicMock(return_value=mock_chroma_collection)
    return client


@pytest.fixture
def retriever(mock_chroma_client):
    """RAGRetriever with ChromaDB mocked out at the _get_client level."""
    from llm.rag import RAGRetriever
    r = RAGRetriever()
    # Inject mock client so _get_client() returns it without hitting the network
    r._client = mock_chroma_client
    r._collections = {}
    return r


# ---------------------------------------------------------------------------
# Collection naming — isolation (C1)
# ---------------------------------------------------------------------------

class TestCollectionNaming:

    def test_collection_name_includes_user_id(self, retriever, mock_chroma_client):
        retriever._get_collection(USER_A)
        call_args = mock_chroma_client.get_or_create_collection.call_args
        name_used = call_args[1].get("name") or call_args[0][0]
        assert f"vector_docs_{USER_A}" == name_used

    def test_different_users_get_different_collections(self, retriever, mock_chroma_client):
        retriever._get_collection(USER_A)
        retriever._get_collection(USER_B)
        calls = mock_chroma_client.get_or_create_collection.call_args_list
        names = [c[1].get("name") or c[0][0] for c in calls]
        assert f"vector_docs_{USER_A}" in names
        assert f"vector_docs_{USER_B}" in names

    def test_collection_cached_after_first_call(self, retriever, mock_chroma_client):
        retriever._get_collection(USER_A)
        retriever._get_collection(USER_A)
        assert mock_chroma_client.get_or_create_collection.call_count == 1

    def test_cache_is_per_user(self, retriever, mock_chroma_client):
        retriever._get_collection(USER_A)
        retriever._get_collection(USER_B)
        assert mock_chroma_client.get_or_create_collection.call_count == 2


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------

class TestIngest:

    @pytest.mark.asyncio
    async def test_ingest_requires_user_id(self, retriever):
        with pytest.raises(TypeError):
            await retriever.ingest(text=SAMPLE_TEXT)  # missing user_id

    @pytest.mark.asyncio
    async def test_ingest_calls_embed_for_each_chunk(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
            result = await retriever.ingest(text=SAMPLE_TEXT, user_id=TEST_USER)
        assert mock_ollama.embed.call_count >= 1  # at least one chunk

    @pytest.mark.asyncio
    async def test_ingest_returns_doc_id_and_chunks(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
            result = await retriever.ingest(text=SAMPLE_TEXT, user_id=TEST_USER, doc_id="doc_42")
        assert result["doc_id"] == "doc_42"
        assert result["chunks_stored"] >= 1

    @pytest.mark.asyncio
    async def test_ingest_adds_to_correct_collection(self, retriever, mock_chroma_client):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            await retriever.ingest(text=SAMPLE_TEXT, user_id=TEST_USER, doc_id="doc_1")
        name_used = mock_chroma_client.get_or_create_collection.call_args[1].get("name") \
                    or mock_chroma_client.get_or_create_collection.call_args[0][0]
        assert name_used == f"vector_docs_{TEST_USER}"

    @pytest.mark.asyncio
    async def test_ingest_user_a_does_not_touch_user_b_collection(self, retriever, mock_chroma_client):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            await retriever.ingest(text=SAMPLE_TEXT, user_id=USER_A, doc_id="doc_1")
        names = [c[1].get("name") or c[0][0]
                 for c in mock_chroma_client.get_or_create_collection.call_args_list]
        assert not any(USER_B in n for n in names)

    @pytest.mark.asyncio
    async def test_ingest_autogenerates_doc_id(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            result = await retriever.ingest(text=SAMPLE_TEXT, user_id=TEST_USER)
        assert result["doc_id"]  # not empty


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:

    @pytest.mark.asyncio
    async def test_retrieve_requires_user_id(self, retriever):
        with pytest.raises(TypeError):
            await retriever.retrieve(query="revenue last quarter")

    @pytest.mark.asyncio
    async def test_retrieve_returns_string(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2])
            result = await retriever.retrieve(query="revenue", user_id=TEST_USER)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_retrieve_embeds_query(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1, 0.2])
            await retriever.retrieve(query="revenue last quarter", user_id=TEST_USER)
        mock_ollama.embed.assert_called_once_with("revenue last quarter")

    @pytest.mark.asyncio
    async def test_retrieve_queries_correct_collection(self, retriever, mock_chroma_client):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            await retriever.retrieve(query="expenses", user_id=TEST_USER)
        name_used = mock_chroma_client.get_or_create_collection.call_args[1].get("name") \
                    or mock_chroma_client.get_or_create_collection.call_args[0][0]
        assert name_used == f"vector_docs_{TEST_USER}"

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_when_no_docs(self, retriever, mock_chroma_collection):
        mock_chroma_collection.count.return_value = 0
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            result = await retriever.retrieve(query="revenue", user_id=TEST_USER)
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_isolation_users_query_own_collections(self, retriever, mock_chroma_client):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            await retriever.retrieve(query="profit", user_id=USER_A)
            await retriever.retrieve(query="loss",   user_id=USER_B)
        names = [c[1].get("name") or c[0][0]
                 for c in mock_chroma_client.get_or_create_collection.call_args_list]
        assert f"vector_docs_{USER_A}" in names
        assert f"vector_docs_{USER_B}" in names


# ---------------------------------------------------------------------------
# delete_doc()
# ---------------------------------------------------------------------------

class TestDeleteDoc:

    @pytest.mark.asyncio
    async def test_delete_doc_requires_user_id(self, retriever):
        with pytest.raises(TypeError):
            await retriever.delete_doc(doc_id="doc_1")

    @pytest.mark.asyncio
    async def test_delete_doc_returns_chunks_removed(self, retriever):
        result = await retriever.delete_doc(doc_id="doc_1", user_id=TEST_USER)
        assert "chunks_removed" in result
        assert "doc_id" in result

    @pytest.mark.asyncio
    async def test_delete_doc_only_touches_own_collection(self, retriever, mock_chroma_client):
        await retriever.delete_doc(doc_id="doc_1", user_id=USER_A)
        name_used = mock_chroma_client.get_or_create_collection.call_args[1].get("name") \
                    or mock_chroma_client.get_or_create_collection.call_args[0][0]
        assert name_used == f"vector_docs_{USER_A}"


# ---------------------------------------------------------------------------
# list_docs()
# ---------------------------------------------------------------------------

class TestListDocs:

    @pytest.mark.asyncio
    async def test_list_docs_requires_user_id(self, retriever):
        with pytest.raises(TypeError):
            await retriever.list_docs()

    @pytest.mark.asyncio
    async def test_list_docs_returns_list(self, retriever):
        result = await retriever.list_docs(user_id=TEST_USER)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_docs_returns_correct_doc_ids(self, retriever):
        result = await retriever.list_docs(user_id=TEST_USER)
        doc_ids = [item["doc_id"] for item in result]
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids

    @pytest.mark.asyncio
    async def test_list_docs_deduplicates(self, retriever):
        """Each doc_id should appear only once even with multiple chunks."""
        result = await retriever.list_docs(user_id=TEST_USER)
        doc_ids = [item["doc_id"] for item in result]
        assert len(doc_ids) == len(set(doc_ids))

    @pytest.mark.asyncio
    async def test_list_docs_isolation(self, retriever, mock_chroma_client):
        await retriever.list_docs(user_id=USER_A)
        name_used = mock_chroma_client.get_or_create_collection.call_args[1].get("name") \
                    or mock_chroma_client.get_or_create_collection.call_args[0][0]
        assert name_used == f"vector_docs_{USER_A}"


# ---------------------------------------------------------------------------
# asyncio.to_thread wrapping (H4)
# ---------------------------------------------------------------------------

class TestAsyncWrapping:

    @pytest.mark.asyncio
    async def test_retrieve_completes_without_blocking(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            result = await retriever.retrieve(user_id=TEST_USER, query="test query")
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_retrieves_different_users(self, retriever):
        with patch("llm.rag.ollama_client") as mock_ollama:
            mock_ollama.embed = AsyncMock(return_value=[0.1])
            results = await asyncio.gather(
                retriever.retrieve(user_id=USER_A, query="revenue"),
                retriever.retrieve(user_id=USER_B, query="expenses"),
            )
        assert len(results) == 2
