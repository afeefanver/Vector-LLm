"""
tests/test_router.py
Tests for LLMRouter (class name in source) — keyword fast-path (H2).

Key signatures confirmed from source:
  - Class:         LLMRouter   (singleton exported as `router`)
  - RouteResult fields: intent, confidence, refined_query, original, fast_path
  - Keyword fn:    _keyword_classify(query) — module-level function
  - Intent enum:   Intent.DASHBOARD / DECISION / QUESTION / FORECAST / SUMMARY
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# RouteResult field contract
# ---------------------------------------------------------------------------

class TestRouteResultFields:

    def test_has_fast_path_field(self):
        from llm.router import RouteResult, Intent
        r = RouteResult(intent=Intent.DASHBOARD, confidence=0.92,
                        refined_query="show chart", original="show chart", fast_path=True)
        assert hasattr(r, "fast_path")

    def test_fast_path_is_bool(self):
        from llm.router import RouteResult, Intent
        r = RouteResult(intent=Intent.DASHBOARD, confidence=0.92,
                        refined_query="show chart", original="show chart", fast_path=True)
        assert r.fast_path is True

    def test_all_five_fields_present(self):
        from llm.router import RouteResult
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(RouteResult)}
        assert "fast_path"     in field_names
        assert "intent"        in field_names
        assert "confidence"    in field_names
        assert "refined_query" in field_names
        assert "original"      in field_names

    def test_intent_enum_values(self):
        from llm.router import Intent
        assert Intent.DASHBOARD.value == "dashboard"
        assert Intent.DECISION.value  == "decision"
        assert Intent.QUESTION.value  == "question"
        assert Intent.FORECAST.value  == "forecast"
        assert Intent.SUMMARY.value   == "summary"


# ---------------------------------------------------------------------------
# _keyword_classify() — pure module-level function
# ---------------------------------------------------------------------------

class TestKeywordClassify:

    # --- DASHBOARD ---
    @pytest.mark.parametrize("query", [
        "show me a bar chart of monthly revenue",
        "plot expenses over time",
        "visualise profit margins",
        "chart my inventory turnover",
        "dashboard for last quarter",
    ])
    def test_dashboard_keywords(self, query):
        from llm.router import _keyword_classify, Intent
        result = _keyword_classify(query)
        assert result is not None
        assert result.intent == Intent.DASHBOARD
        assert result.fast_path is True
        assert result.confidence >= 0.85

    # --- DECISION ---
    @pytest.mark.parametrize("query", [
        "should we expand to Mumbai?",
        "should I hire another salesperson?",
        "recommend a strategy for Q3",
        "advise me on cutting costs",
        "is it worth investing in the new factory?",
    ])
    def test_decision_keywords(self, query):
        from llm.router import _keyword_classify, Intent
        result = _keyword_classify(query)
        assert result is not None
        assert result.intent == Intent.DECISION
        assert result.fast_path is True

    # --- FORECAST ---
    @pytest.mark.parametrize("query", [
        "forecast revenue for next quarter",
        "predict sales for next month",
        "what is the sales projection for next year?",
    ])
    def test_forecast_keywords(self, query):
        from llm.router import _keyword_classify, Intent
        result = _keyword_classify(query)
        assert result is not None
        assert result.intent == Intent.FORECAST
        assert result.fast_path is True

    # --- SUMMARY ---
    @pytest.mark.parametrize("query", [
        "summarise last month's performance",
        "give me an overview of Q2",
        "explain this data to me",
    ])
    def test_summary_keywords(self, query):
        from llm.router import _keyword_classify, Intent
        result = _keyword_classify(query)
        assert result is not None
        assert result.intent == Intent.SUMMARY
        assert result.fast_path is True

    # --- QUESTION ---
    @pytest.mark.parametrize("query", [
        "what was revenue in March?",
        "how much did we spend on salaries?",
        "what is the total revenue?",
    ])
    def test_question_keywords(self, query):
        from llm.router import _keyword_classify, Intent
        result = _keyword_classify(query)
        assert result is not None
        assert result.intent == Intent.QUESTION
        assert result.fast_path is True

    # --- Ambiguous → None ---
    @pytest.mark.parametrize("query", [
        "hmmm",
        "okay",
        "not sure",
    ])
    def test_ambiguous_returns_none(self, query):
        from llm.router import _keyword_classify
        result = _keyword_classify(query)
        assert result is None

    # --- Confidence levels ---
    def test_strong_keyword_confidence(self):
        from llm.router import _keyword_classify
        result = _keyword_classify("show me a bar chart of revenue")
        assert result is not None
        assert result.confidence == 0.92

    def test_medium_keyword_confidence(self):
        from llm.router import _keyword_classify
        result = _keyword_classify("can you show me the dashboard")
        assert result is not None
        assert result.confidence == 0.85

    # --- Case insensitivity ---
    def test_keywords_are_case_insensitive(self):
        from llm.router import _keyword_classify
        lower = _keyword_classify("show me a bar chart")
        upper = _keyword_classify("SHOW ME A BAR CHART")
        assert lower is not None
        assert upper is not None
        assert lower.intent == upper.intent

    # --- fast_path always True when result returned ---
    def test_keyword_classify_always_sets_fast_path_true(self):
        from llm.router import _keyword_classify
        result = _keyword_classify("show me a bar chart")
        assert result is not None
        assert result.fast_path is True

    # --- original preserved ---
    def test_original_query_preserved(self):
        from llm.router import _keyword_classify
        q = "Show me a Bar Chart"
        result = _keyword_classify(q)
        assert result is not None
        assert result.original == q


# ---------------------------------------------------------------------------
# LLMRouter.route() — full method
# ---------------------------------------------------------------------------

class TestRouteFullMethod:

    @pytest.fixture
    def router_instance(self):
        from llm.router import LLMRouter
        return LLMRouter()

    @pytest.mark.asyncio
    async def test_fast_path_skips_llm(self, router_instance):
        from unittest.mock import patch, AsyncMock
        with patch("llm.router.ollama_client") as mock_ollama:
            mock_ollama.complete = AsyncMock(return_value='{"intent":"question","confidence":0.8,"refined_query":"test"}')
            result = await router_instance.route("show me a bar chart of revenue")
        assert result.fast_path is True
        mock_ollama.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_fallback_sets_fast_path_false(self, router_instance):
        from unittest.mock import patch, AsyncMock
        with patch("llm.router.ollama_client") as mock_ollama:
            mock_ollama.complete = AsyncMock(
                return_value='{"intent":"question","confidence":0.75,"refined_query":"ambiguous query"}'
            )
            result = await router_instance.route("hmmm not sure")
        assert result.fast_path is False

    @pytest.mark.asyncio
    async def test_route_returns_route_result(self, router_instance):
        from llm.router import RouteResult
        from unittest.mock import patch, AsyncMock
        with patch("llm.router.ollama_client") as mock_ollama:
            mock_ollama.complete = AsyncMock(
                return_value='{"intent":"question","confidence":0.9,"refined_query":"test"}'
            )
            result = await router_instance.route("what was revenue?")
        assert isinstance(result, RouteResult)

    @pytest.mark.asyncio
    async def test_route_intent_is_valid_enum(self, router_instance):
        from llm.router import Intent
        from unittest.mock import patch, AsyncMock
        with patch("llm.router.ollama_client") as mock_ollama:
            mock_ollama.complete = AsyncMock(
                return_value='{"intent":"question","confidence":0.9,"refined_query":"test"}'
            )
            result = await router_instance.route("what was revenue in March?")
        assert isinstance(result.intent, Intent)

    @pytest.mark.asyncio
    async def test_llm_fallback_on_bad_json_returns_question(self, router_instance):
        """If LLM returns garbage, router should fall back gracefully."""
        from llm.router import Intent
        from unittest.mock import patch, AsyncMock
        with patch("llm.router.ollama_client") as mock_ollama:
            mock_ollama.complete = AsyncMock(return_value="this is not json at all!!")
            result = await router_instance.route("completely ambiguous input blarg")
        # Graceful fallback — must not raise, and intent must be valid
        assert isinstance(result.intent, Intent)

    @pytest.mark.asyncio
    async def test_fast_path_majority_of_typical_queries(self, router_instance):
        """At least 7/10 representative queries should hit fast path."""
        from unittest.mock import patch, AsyncMock
        queries = [
            "show me a bar chart of revenue",
            "plot expenses over time",
            "should I hire more staff?",
            "summarise Q2 performance",
            "forecast next quarter sales",
            "what was total revenue in March?",
            "recommend a strategy",
            "hmmm",
            "okay",
            "not sure what",
        ]
        fast = 0
        with patch("llm.router.ollama_client") as mock_ollama:
            mock_ollama.complete = AsyncMock(
                return_value='{"intent":"question","confidence":0.7,"refined_query":"q"}'
            )
            for q in queries:
                result = await router_instance.route(q)
                if result.fast_path:
                    fast += 1
        assert fast >= 7
