"""
llm/router.py
-------------
Reads every incoming query, classifies its intent, and routes it
to the right handler. This is the brain of the LLM layer.

Intent categories:
  dashboard  — user wants a chart or visual
  decision   — user wants a recommendation or strategy
  question   — user wants a factual answer from their data
  forecast   — user wants a prediction / trend
  summary    — user wants a plain-language summary
"""

from enum import Enum
from dataclasses import dataclass
from llm.ollama_client import OllamaClient
from core.config import settings


class Intent(str, Enum):
    DASHBOARD = "dashboard"
    DECISION  = "decision"
    QUESTION  = "question"
    FORECAST  = "forecast"
    SUMMARY   = "summary"


@dataclass
class RouteResult:
    intent: Intent
    confidence: float
    refined_query: str          # cleaned-up version sent to the handler


CLASSIFY_PROMPT = """You are a query classifier for a data analytics product.

Given the user query below, respond with ONLY a JSON object — no explanation, no markdown.

{{
  "intent": one of ["dashboard", "decision", "question", "forecast", "summary"],
  "confidence": float between 0 and 1,
  "refined_query": the query rewritten clearly and concisely
}}

Rules:
- dashboard: user wants a chart, graph, visualisation, or table
- decision: user wants a recommendation, should/shouldn't, best option, strategy
- question: user wants a specific fact or number from their data
- forecast: user wants a future prediction, trend, or projection
- summary: user wants a high-level overview or explanation

User query: "{query}"
"""


class LLMRouter:
    def __init__(self):
        self.client = OllamaClient()

    async def route(self, query: str) -> RouteResult:
        prompt = CLASSIFY_PROMPT.format(query=query)
        raw = await self.client.complete(prompt, max_tokens=120)

        try:
            import json
            data = json.loads(raw.strip())
            return RouteResult(
                intent=Intent(data["intent"]),
                confidence=float(data["confidence"]),
                refined_query=data.get("refined_query", query),
            )
        except Exception:
            # Fallback — default to question intent
            return RouteResult(
                intent=Intent.QUESTION,
                confidence=0.5,
                refined_query=query,
            )


# Module-level singleton
router = LLMRouter()
