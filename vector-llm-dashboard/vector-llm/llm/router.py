"""
llm/router.py
=============
Reads every incoming query and decides what kind of task it is.
This runs FIRST before any other engine is called.

WHY THIS EXISTS:
  The team sends all queries to your microservice.
  You need to know: does the user want a chart? a decision? a plain answer?
  The router figures that out using the LLM itself, then returns a label.

INTENT TYPES:
  dashboard  → user wants a chart or visual ("show me a bar chart of sales")
  decision   → user wants a recommendation ("should we expand to Mumbai?")
  question   → user wants a fact from their data ("what was revenue in March?")
  forecast   → user wants a future prediction ("what will sales be next quarter?")
  summary    → user wants a plain overview ("summarise this CSV")

HOW IT WORKS:
  1. Send the user's query to Mistral with a classification prompt
  2. Mistral returns a JSON object with intent + confidence + cleaned query
  3. Router returns a RouteResult dataclass
  4. The main API uses RouteResult.intent to call the right engine
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
    confidence:    float    # how sure the model is (0.0 – 1.0)
    refined_query: str      # cleaned-up version of the original query
    original:      str      # original query unchanged


# ------------------------------------------------------------------
# Prompt
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

        Example:
            from llm.router import router
            result = await router.route("show me monthly revenue as a bar chart")
            # result.intent     → Intent.DASHBOARD
            # result.confidence → 0.95
            # result.refined_query → "Show monthly revenue as a bar chart"
        """
        prompt = CLASSIFY_PROMPT.format(query=query.strip())

        try:
            raw = await ollama_client.complete(prompt, max_tokens=150)
            data = self._parse_json(raw)

            return RouteResult(
                intent        = Intent(data["intent"]),
                confidence    = float(data.get("confidence", 0.7)),
                refined_query = data.get("refined_query", query),
                original      = query,
            )

        except Exception as e:
            # If anything goes wrong, fall back to QUESTION intent
            # so the user still gets a response rather than a crash
            print(f"[router] fallback triggered: {e}")
            return RouteResult(
                intent        = Intent.QUESTION,
                confidence    = 0.4,
                refined_query = query,
                original      = query,
            )

    def _parse_json(self, raw: str) -> dict:
        """
        Safely extract JSON from the model's response.
        Handles cases where the model wraps output in markdown fences.
        """
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError(f"Could not parse JSON from model response: {raw[:200]}")


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.router import router, Intent
# ------------------------------------------------------------------
router = LLMRouter()
