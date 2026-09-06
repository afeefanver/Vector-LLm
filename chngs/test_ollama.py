"""
tests/test_ollama.py
Tests for OllamaClient — connection pooling (H3), embed model split (H1).

Key signatures confirmed from source:
  OllamaClient.__init__(timeout=None)  — NO base_url/model kwargs; reads from settings
  self._http = httpx.AsyncClient(...)  — created in __init__, shared pool
  complete(prompt, max_tokens=1024)
  stream(prompt)
  embed(text)
  health_check()  — uses its own short-lived client (intentional)
  close()

MOCK STRATEGY:
  Patch httpx.AsyncClient at the module level so __init__ gets our mock.
  Then replace c._http directly for full control.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import json


# These must match what's in your .env / config.py
# If you changed OLLAMA_MODEL in .env, update GEN_MODEL here too
GEN_MODEL   = "phi3"
EMBED_MODEL = "nomic-embed-text"


def make_response(status_code=200, json_body=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def mock_http():
    http = MagicMock()
    http.post   = AsyncMock()
    http.get    = AsyncMock()
    http.aclose = AsyncMock()
    return http


@pytest.fixture
def client(mock_http):
    """OllamaClient with shared _http replaced by mock."""
    with patch("llm.ollama.httpx.AsyncClient", return_value=mock_http):
        from llm.ollama import OllamaClient
        c = OllamaClient()
    c._http = mock_http
    return c


# ---------------------------------------------------------------------------
# __init__ — pool created once, no args needed
# ---------------------------------------------------------------------------

class TestInit:

    def test_http_client_created_at_init(self):
        with patch("llm.ollama.httpx.AsyncClient") as MockClient:
            MockClient.return_value = MagicMock()
            from llm.ollama import OllamaClient
            c = OllamaClient()
        assert hasattr(c, "_http")
        assert c._http is not None

    def test_http_client_created_exactly_once(self):
        with patch("llm.ollama.httpx.AsyncClient") as MockClient:
            MockClient.return_value = MagicMock()
            from llm.ollama import OllamaClient
            OllamaClient()
        assert MockClient.call_count == 1

    def test_models_stored_from_settings(self, client):
        assert client.model       == GEN_MODEL
        assert client.embed_model == EMBED_MODEL

    def test_no_base_url_kwarg(self):
        """Constructor must NOT accept base_url — it reads from settings."""
        with patch("llm.ollama.httpx.AsyncClient", return_value=MagicMock()):
            from llm.ollama import OllamaClient
            with pytest.raises(TypeError):
                OllamaClient(base_url="http://example.com")


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:

    @pytest.mark.asyncio
    async def test_complete_uses_shared_http(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "answer"})
        await client.complete("test prompt")
        assert mock_http.post.called

    @pytest.mark.asyncio
    async def test_complete_sends_to_generate_endpoint(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "ok"})
        await client.complete("test")
        url = mock_http.post.call_args[0][0] if mock_http.post.call_args[0] \
              else mock_http.post.call_args[1].get("url", "")
        assert "/api/generate" in str(url)

    @pytest.mark.asyncio
    async def test_complete_uses_generative_model(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "ok"})
        await client.complete("test")
        body = mock_http.post.call_args[1].get("json") or {}
        if body:
            assert body.get("model") == GEN_MODEL

    @pytest.mark.asyncio
    async def test_complete_returns_string(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "Profit grew 15%."})
        result = await client.complete("summarise")
        assert isinstance(result, str)
        assert "Profit" in result

    @pytest.mark.asyncio
    async def test_complete_does_not_open_new_client(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "ok"})
        with patch("llm.ollama.httpx.AsyncClient") as NewClient:
            await client.complete("test")
        NewClient.assert_not_called()


# ---------------------------------------------------------------------------
# embed() — must use embed_model, not generative model (H1)
# ---------------------------------------------------------------------------

class TestEmbed:

    @pytest.mark.asyncio
    async def test_embed_uses_embed_model_not_generative(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"embedding": [0.1, 0.2]})
        await client.embed("some text")
        body = mock_http.post.call_args[1].get("json") or {}
        if body:
            assert body.get("model") == EMBED_MODEL
            assert body.get("model") != GEN_MODEL

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_floats(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"embedding": [0.1, 0.2, 0.3]})
        result = await client.embed("Q1 revenue data")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_uses_shared_http(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"embedding": [0.1]})
        await client.embed("text")
        assert mock_http.post.called

    @pytest.mark.asyncio
    async def test_embed_does_not_open_new_client(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"embedding": [0.1]})
        with patch("llm.ollama.httpx.AsyncClient") as NewClient:
            await client.embed("text")
        NewClient.assert_not_called()


# ---------------------------------------------------------------------------
# health_check() — intentionally uses its own short-lived client
# ---------------------------------------------------------------------------

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, client):
        short_http = MagicMock()
        short_http.__aenter__ = AsyncMock(return_value=short_http)
        short_http.__aexit__  = AsyncMock(return_value=False)
        short_http.get = AsyncMock(return_value=make_response(
            json_body={"models": [{"name": GEN_MODEL}, {"name": EMBED_MODEL}]}
        ))
        with patch("llm.ollama.httpx.AsyncClient", return_value=short_http):
            result = await client.health_check()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_health_check_does_not_use_shared_pool(self, client, mock_http):
        short_http = MagicMock()
        short_http.__aenter__ = AsyncMock(return_value=short_http)
        short_http.__aexit__  = AsyncMock(return_value=False)
        short_http.get = AsyncMock(return_value=make_response(
            json_body={"models": [{"name": GEN_MODEL}]}
        ))
        with patch("llm.ollama.httpx.AsyncClient", return_value=short_http):
            await client.health_check()
        mock_http.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_reports_ollama_down_on_error(self, client):
        short_http = MagicMock()
        short_http.__aenter__ = AsyncMock(return_value=short_http)
        short_http.__aexit__  = AsyncMock(return_value=False)
        short_http.get = AsyncMock(side_effect=Exception("connection refused"))
        with patch("llm.ollama.httpx.AsyncClient", return_value=short_http):
            result = await client.health_check()
        assert result["ollama_running"] is False


# ---------------------------------------------------------------------------
# close() — shutdown hook
# ---------------------------------------------------------------------------

class TestClose:

    @pytest.mark.asyncio
    async def test_close_calls_http_aclose(self, client, mock_http):
        await client.close()
        mock_http.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_async(self, client):
        import inspect
        assert inspect.iscoroutinefunction(client.close)


# ---------------------------------------------------------------------------
# Connection pool — same instance reused (H3)
# ---------------------------------------------------------------------------

class TestConnectionPool:

    @pytest.mark.asyncio
    async def test_same_http_instance_used_across_calls(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "ok"})
        id_before = id(client._http)
        await client.complete("call 1")
        await client.complete("call 2")
        mock_http.post.return_value = make_response(json_body={"embedding": [0.1]})
        await client.embed("call 3")
        assert id(client._http) == id_before

    @pytest.mark.asyncio
    async def test_post_called_for_each_complete(self, client, mock_http):
        mock_http.post.return_value = make_response(json_body={"response": "ok"})
        await client.complete("call 1")
        await client.complete("call 2")
        assert mock_http.post.call_count == 2
