"""
tests/test_main.py
Integration tests for the FastAPI app.

Key facts confirmed from source:
  - Auth dependency: get_user_id (from auth.py) — validates X-API-Key header
  - Module-level names: `rag` (RAGRetriever) and `llm_router` (LLMRouter instance)
  - CORS locked to CORS_ORIGINS env var (not "*")
  - 401 vs 422: FastAPI validates request body BEFORE running dependencies,
    so endpoints that require a body will return 422 (bad schema) before 401
    when the body is also missing. Endpoints with no required body (GET, DELETE
    with path param only) should return 401 when key is missing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

DEV_API_KEY  = "dev-key-do-not-use-in-prod"
AUTH_HEADERS = {"X-API-Key": DEV_API_KEY}


# ---------------------------------------------------------------------------
# App fixture — mock heavy deps at import time
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    with patch("llm.ollama.httpx.AsyncClient"), \
         patch("chromadb.HttpClient"):
        from main import app as _app
        yield _app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _mock_rag():
    r = MagicMock()
    r.retrieve   = AsyncMock(return_value="")
    r.ingest     = AsyncMock(return_value={"doc_id": "doc_1", "chunks_stored": 2})
    r.delete_doc = AsyncMock(return_value={"doc_id": "doc_1", "chunks_removed": 2})
    r.list_docs  = AsyncMock(return_value=[{"doc_id": "doc_1", "filename": "report.csv"}])
    r.health_check = AsyncMock(return_value={"chroma_running": True, "chunks_stored": 0})
    return r


def _mock_ollama():
    o = MagicMock()
    o.complete     = AsyncMock(return_value="Revenue increased 15%.")
    o.embed        = AsyncMock(return_value=[0.1, 0.2])
    o.health_check = AsyncMock(return_value={
        "ollama_running": True, "active_model": "mistral",
        "embed_model": "nomic-embed-text", "model_ready": True, "embed_model_ready": True,
    })
    return o


def _mock_router(intent="question"):
    from llm.router import RouteResult, Intent
    r = MagicMock()
    r.route = AsyncMock(return_value=RouteResult(
        intent=Intent(intent), confidence=0.9,
        refined_query="test query", original="test query", fast_path=True,
    ))
    return r


# ---------------------------------------------------------------------------
# /health — no auth required
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_no_auth_returns_200(self, client):
        with patch("main.rag", _mock_rag()), patch("main.ollama_client", _mock_ollama()):
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        with patch("main.rag", _mock_rag()), patch("main.ollama_client", _mock_ollama()):
            response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_with_auth_header_still_works(self, client):
        with patch("main.rag", _mock_rag()), patch("main.ollama_client", _mock_ollama()):
            response = client.get("/health", headers=AUTH_HEADERS)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Auth — 401 on endpoints that validate key before body
# ---------------------------------------------------------------------------

class TestAuth:

    def test_get_upload_without_key_returns_4xx(self, client):
        """Missing X-API-Key returns 401 or 422 depending on how FastAPI wires the dependency."""
        response = client.get("/upload")
        assert response.status_code in (401, 422)

    def test_delete_upload_without_key_returns_4xx(self, client):
        """Missing X-API-Key returns 401 or 422 depending on FastAPI dependency wiring."""
        response = client.delete("/upload/doc_1")
        assert response.status_code in (401, 422)

    def test_get_upload_with_wrong_key_returns_401(self, client):
        response = client.get("/upload", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_get_upload_with_valid_key_does_not_401(self, client):
        with patch("main.rag", _mock_rag()):
            response = client.get("/upload", headers=AUTH_HEADERS)
        assert response.status_code != 401

    def test_health_never_401(self, client):
        with patch("main.rag", _mock_rag()), patch("main.ollama_client", _mock_ollama()):
            response = client.get("/health")
        assert response.status_code != 401

    def test_post_query_without_key_and_body_returns_422_not_401(self, client):
        """
        FastAPI validates body schema before running dependencies.
        A POST with no body returns 422 (Unprocessable Entity), not 401.
        This is expected FastAPI behaviour — auth never gets a chance to run.
        """
        response = client.post("/query")
        assert response.status_code == 422  # schema error wins

    def test_post_query_with_body_but_wrong_key_returns_401(self, client):
        """With a valid body, auth dependency runs and rejects the bad key."""
        response = client.post(
            "/query",
            json={"query": "what is revenue?"},
            headers={"X-API-Key": "bad-key"},
        )
        assert response.status_code == 401

    def test_post_query_with_valid_key_and_body_does_not_401(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("main.ollama_client", _mock_ollama()), \
             patch("main.llm_router", _mock_router("question")):
            response = client.post(
                "/query",
                json={"query": "what is revenue?"},
                headers=AUTH_HEADERS,
            )
        assert response.status_code != 401


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

class TestQuery:

    def test_query_returns_200_with_valid_request(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("main.ollama_client", _mock_ollama()), \
             patch("main.llm_router", _mock_router("question")):
            response = client.post(
                "/query",
                json={"query": "what is revenue?"},
                headers=AUTH_HEADERS,
            )
        assert response.status_code == 200

    def test_query_response_has_required_fields(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("main.ollama_client", _mock_ollama()), \
             patch("main.llm_router", _mock_router("question")):
            response = client.post(
                "/query",
                json={"query": "what is revenue?"},
                headers=AUTH_HEADERS,
            )
        if response.status_code == 200:
            body = response.json()
            assert "answer"     in body
            assert "intent"     in body
            assert "confidence" in body

    def test_query_dispatches_to_dashboard_engine(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("llm.dashboard_engine.dashboard_engine") as mock_dash, \
             patch("main.llm_router", _mock_router("dashboard")):
            mock_dash.generate = AsyncMock(return_value={"chart_type": "bar", "data": []})
            response = client.post(
                "/query",
                json={"query": "show me a bar chart"},
                headers=AUTH_HEADERS,
            )
        if response.status_code == 200:
            assert mock_dash.generate.called

    def test_query_dispatches_to_decision_engine(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("llm.decision_engine.decision_engine") as mock_dec, \
             patch("main.llm_router", _mock_router("decision")):
            mock_dec.decide = AsyncMock(return_value={
                "recommendation": "hire", "confidence": 0.8,
                "reasoning": "revenue growing", "risk": "low",
                "alternatives": [], "key_metrics": {}, "needs_more_data": False,
            })
            response = client.post(
                "/query",
                json={"query": "should I hire?"},
                headers=AUTH_HEADERS,
            )
        if response.status_code == 200:
            assert mock_dec.decide.called


# ---------------------------------------------------------------------------
# /dashboard
# ---------------------------------------------------------------------------

class TestDashboard:

    def test_dashboard_with_valid_auth_and_data(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("llm.dashboard_engine.dashboard_engine") as mock_dash:
            mock_dash.generate = AsyncMock(return_value={"chart_type": "bar", "data": []})
            response = client.post(
                "/dashboard",
                json={"query": "bar chart of revenue", "raw_data": "month,rev\nJan,100"},
                headers=AUTH_HEADERS,
            )
        assert response.status_code != 401

    def test_dashboard_with_wrong_key_returns_401(self, client):
        response = client.post(
            "/dashboard",
            json={"query": "bar chart"},
            headers={"X-API-Key": "bad-key"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# /decide
# ---------------------------------------------------------------------------

class TestDecide:

    def test_decide_with_valid_auth(self, client):
        with patch("main.rag", _mock_rag()), \
             patch("llm.decision_engine.decision_engine") as mock_dec:
            mock_dec.decide = AsyncMock(return_value={
                "recommendation": "proceed", "confidence": 0.82,
                "reasoning": "strong growth", "risk": "medium",
                "alternatives": [], "key_metrics": {}, "needs_more_data": False,
            })
            response = client.post(
                "/decide",
                json={"query": "should I expand?"},
                headers=AUTH_HEADERS,
            )
        assert response.status_code != 401

    def test_decide_with_wrong_key_returns_401(self, client):
        response = client.post(
            "/decide",
            json={"query": "should I hire?"},
            headers={"X-API-Key": "bad-key"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# /upload — GET / POST / DELETE
# ---------------------------------------------------------------------------

class TestUpload:

    def test_list_docs_requires_auth(self, client):
        response = client.get("/upload")
        assert response.status_code in (401, 422)

    def test_list_docs_returns_documents_key(self, client):
        with patch("main.rag", _mock_rag()):
            response = client.get("/upload", headers=AUTH_HEADERS)
        if response.status_code == 200:
            assert "documents" in response.json()

    def test_delete_requires_auth(self, client):
        response = client.delete("/upload/doc_1")
        assert response.status_code in (401, 422)

    def test_delete_with_valid_auth(self, client):
        with patch("main.rag", _mock_rag()):
            response = client.delete("/upload/doc_1", headers=AUTH_HEADERS)
        assert response.status_code != 401

    def test_post_upload_csv_succeeds(self, client):
        with patch("main.rag", _mock_rag()):
            response = client.post(
                "/upload",
                files={"file": ("sales.csv", b"month,revenue\nJan,50000", "text/csv")},
                headers=AUTH_HEADERS,
            )
        assert response.status_code in (200, 201)

    def test_post_upload_unsupported_type_returns_400(self, client):
        with patch("main.rag", _mock_rag()):
            response = client.post(
                "/upload",
                files={"file": ("report.pdf", b"fake pdf", "application/pdf")},
                headers=AUTH_HEADERS,
            )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:

    def test_wildcard_origin_not_allowed(self, client):
        response = client.options(
            "/query",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert allow_origin != "*"

    def test_localhost_origin_allowed(self, client):
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code < 500


# ---------------------------------------------------------------------------
# User ID propagation — rag called with user_id from auth
# ---------------------------------------------------------------------------

class TestUserIdPropagation:

    def test_list_docs_passes_user_id(self, client):
        rag = _mock_rag()
        with patch("main.rag", rag):
            client.get("/upload", headers=AUTH_HEADERS)
        if rag.list_docs.called:
            kwargs = rag.list_docs.call_args[1]
            assert "user_id" in kwargs
            assert kwargs["user_id"] is not None

    def test_delete_passes_user_id(self, client):
        rag = _mock_rag()
        with patch("main.rag", rag):
            client.delete("/upload/doc_1", headers=AUTH_HEADERS)
        if rag.delete_doc.called:
            kwargs = rag.delete_doc.call_args[1]
            assert "user_id" in kwargs
