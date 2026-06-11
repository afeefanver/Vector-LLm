"""
llm/router.py
=============
Reads every incoming query and decides what kind of task it is.
This runs FIRST before any other engine is called.

WHY THIS EXISTS:
  The team sends all queries to your microservice.
  You need to know: does the user want a chart? a decision? a plain answer?
  The router figures that out, then returns a label.

INTENT TYPES:
  dashboard  -> user wants a chart or visual ("show me a bar chart of sales")
  decision   -> user wants a recommendation ("should we expand to Mumbai?")
  question   -> user wants a fact from their data ("what was revenue in March?")
  forecast   -> user wants a future prediction ("what will sales be next quarter?")
  summary    -> user wants a plain overview ("summarise this CSV")

HOW IT WORKS (H2 fix — two-stage classification):
  Stage 1 — keyword fast-path (runs in ~0ms, no LLM call):
    ~90% of queries contain obvious signal words.
    If a match is found with confidence >= 0.85, return immediately.
    No Ollama call, no latency.

  Stage 2 — LLM classification (runs only for ambiguous queries):
    If Stage 1 finds no clear match, fall through to Mistral.
    Same as before, but now only triggered ~10% of the time.

  This reduces median router latency from ~1-3s to ~0ms for most queries.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum

from llm.ollama import ollama_client


# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------

class Intent(str, Enum):
    DASHBOARD = "dashboard"
    DECISION  = "decision"
    QUESTION  = "question"
    FORECAST  = "forecast"
    SUMMARY   = "summary"


@dataclass
class RouteResult:
    intent:        Intent   # what kind of task this is
    confidence:    float    # how sure the classifier is (0.0 - 1.0)
    refined_query: str      # cleaned-up version of the original query
    original:      str      # original query unchanged
    fast_path:     bool     # True if classified by keyword (no LLM call made)


# ------------------------------------------------------------------
# H2: Keyword fast-path tables
#
# Design rationale:
#   - Words are matched against the lowercased query.
#   - Each intent has two lists:
#       STRONG  (confidence 0.92) — words that almost always mean this intent
#       MEDIUM  (confidence 0.85) — words that usually mean this intent but
#                                   could be ambiguous in isolation
#   - If ANY strong keyword matches -> return at 0.92.
#   - If ANY medium keyword matches -> return at 0.85.
#   - If nothing matches -> fall through to LLM.
#   - Intents are checked in priority order: DASHBOARD, FORECAST, DECISION,
#     SUMMARY, QUESTION. QUESTION is last because words like "what" appear
#     in almost every query and would short-circuit more specific intents.
# ------------------------------------------------------------------

_KEYWORDS: dict[Intent, dict[str, list[str]]] = {
    Intent.DASHBOARD: {
        "strong": [
            "bar chart", "line chart", "pie chart", "scatter plot",
            "histogram", "heatmap", "plotly", "visualise", "visualize",
            "plot this", "draw a", "chart this",
        ],
        "medium": [
            "chart", "graph", "plot", "visual", "show me",
            "display", "dashboard",
        ],
    },
    Intent.FORECAST: {
        "strong": [
            "forecast", "predict", "projection", "next quarter",
            "next month", "next year", "future sales", "trend",
        ],
        "medium": [
            "will it", "will we", "expected", "anticipate",
            "going to be", "estimate future",
        ],
    },
    Intent.DECISION: {
        "strong": [
            "should we", "should i", "recommend", "advise",
            "is it worth", "is it a good idea", "what would you suggest",
            "best option", "best strategy",
        ],
        "medium": [
            "decide", "decision", "strategy", "worth it",
            "expand", "invest", "hire", "launch",
        ],
    },
    Intent.SUMMARY: {
        "strong": [
            "summarise", "summarize", "give me an overview",
            "overview of", "explain this", "what does this data show",
        ],
        "medium": [
            "summary", "overview", "explain", "describe",
            "tell me about",
        ],
    },
    Intent.QUESTION: {
        "strong": [
            "what was", "what is", "how much", "how many",
            "which month", "which product", "top 5", "top 10",
            "highest", "lowest", "average", "total revenue",
            "total sales", "total cost",
        ],
        "medium": [
            "what", "how", "which", "when", "where", "who",
            "number of", "count of",
        ],
    },
}


def _keyword_classify(query: str) -> RouteResult | None:
    """
    Try to classify a query using keyword matching alone.
    Returns a RouteResult if confident, None if the query is ambiguous.

    Checks intents in priority order. Returns on the first strong match,
    then the first medium match across all intents.
    """
    q = query.lower()

    # Pass 1: strong keywords only (confidence 0.92)
    for intent, lists in _KEYWORDS.items():
        if any(kw in q for kw in lists["strong"]):
            return RouteResult(
                intent        = intent,
                confidence    = 0.92,
                refined_query = query.strip(),
                original      = query,
                fast_path     = True,
            )

    # Pass 2: medium keywords (confidence 0.85)
    for intent, lists in _KEYWORDS.items():
        if any(kw in q for kw in lists["medium"]):
            return RouteResult(
                intent        = intent,
                confidence    = 0.85,
                refined_query = query.strip(),
                original      = query,
                fast_path     = True,
            )

    return None  # ambiguous — fall through to LLM


# ------------------------------------------------------------------
# Prompt (used only when fast-path returns None)
# ------------------------------------------------------------------

CLASSIFY_PROMPT = """You are a query classifier for a data analytics product called Vector.

Your only job is to classify the user's query into one of these intents:
  dashboard  - user wants a chart, graph, table, or any visualisation
  decision   - user wants a recommendation, strategy, or yes/no answer
  question   - user wants a specific number or fact from their data
  forecast   - user wants a future prediction or trend
  summary    - user wants a plain-language overview or explanation

Rules:
- Respond ONLY with a JSON object. No explanation. No markdown. No extra text.
- refined_query: rewrite the query to be clear and concise, fix typos
- confidence: your certainty about the intent, between 0.0 and 1.0

Required JSON format:
{{
  "intent": "dashboard" | "decision" | "question" | "forecast" | "summary",
  "confidence": 0.0 to 1.0,
  "refined_query": "cleaned version of the query"
}}

User query: "{query}"
"""


# ------------------------------------------------------------------
# Router class
# ------------------------------------------------------------------

class LLMRouter:

    async def route(self, query: str) -> RouteResult:
        """
        Classify a query and return a RouteResult.

        Tries keyword fast-path first. Falls back to LLM only if ambiguous.

        Example:
            result = await router.route("show me monthly revenue as a bar chart")
            # result.intent     -> Intent.DASHBOARD
            # result.confidence -> 0.92
            # result.fast_path  -> True   (no LLM call made)
        """
        # Stage 1: keyword fast-path (~0ms, no LLM)
        fast = _keyword_classify(query)
        if fast is not None:
            return fast

        # Stage 2: LLM classification (only for ambiguous queries)
        print(f"[router] ambiguous query — falling back to LLM: {query[:80]}")
        return await self._llm_classify(query)

    async def _llm_classify(self, query: str) -> RouteResult:
        """Full LLM classification for queries that stumped the keyword matcher."""
        prompt = CLASSIFY_PROMPT.format(query=query.strip())

        try:
            raw  = await ollama_client.complete(prompt, max_tokens=150)
            data = self._parse_json(raw)

            return RouteResult(
                intent        = Intent(data["intent"]),
                confidence    = float(data.get("confidence", 0.7)),
                refined_query = data.get("refined_query", query),
                original      = query,
                fast_path     = False,
            )

        except Exception as e:
            # If anything goes wrong, fall back to QUESTION intent
            # so the user still gets a response rather than a crash
            print(f"[router] LLM fallback failed: {e}")
            return RouteResult(
                intent        = Intent.QUESTION,
                confidence    = 0.4,
                refined_query = query,
                original      = query,
                fast_path     = False,
            )

    def _parse_json(self, raw: str) -> dict:
        """
        Safely extract JSON from the model's response.
        Handles cases where the model wraps output in markdown fences.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError(f"Could not parse JSON from model response: {raw[:200]}")


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.router import router, Intent
# ------------------------------------------------------------------
router = LLMRouter()
