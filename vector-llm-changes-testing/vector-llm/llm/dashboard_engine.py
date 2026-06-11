"""
llm/dashboard_engine.py
=======================
Takes a user request + their data and returns a Plotly chart spec as JSON.
The LLM decides: chart type, axes, title, series, insight.
Your team's React frontend renders the spec directly — no extra parsing needed.

HOW IT WORKS:
  1. Pull relevant chunks from ChromaDB (via rag.py)
  2. Build a prompt with the context + user query
  3. Mistral returns a Plotly-compatible JSON spec
  4. We validate and score the spec before returning

WHAT THE TEAM RECEIVES:
  {
    "chart_type": "bar",
    "title": "Monthly Revenue 2024",
    "x_label": "Month",
    "y_label": "Revenue (₹)",
    "data": [
      { "x": ["Jan","Feb",...], "y": [52000,61000,...], "type": "bar", "name": "Revenue" }
    ],
    "layout": { "showlegend": false, "colorway": ["#6c63ff"] },
    "insight": "Revenue peaked in December at ₹1,30,000 — a 150% increase from January.",
    "confidence": 0.92
  }

USAGE:
  from llm.dashboard_engine import dashboard_engine
  spec = await dashboard_engine.generate(
      query="show monthly revenue as a bar chart",
      raw_data="month,revenue\\nJan,52000\\n..."   # optional — CSV string
  )
"""

import json
import re
from llm.ollama import ollama_client
from llm.rag import rag
from config import settings


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

DASHBOARD_PROMPT = """You are a data visualisation expert for a product called Vector.

Your job: read the user request and data, then return a Plotly.js chart configuration.

RULES:
- Respond with ONLY a valid JSON object. No explanation. No markdown fences.
- Pick the most appropriate chart_type for the data and request.
- Extract real values from the data context — do not make up numbers.
- Keep titles and labels concise.
- Write one sharp insight sentence explaining what the chart reveals.

chart_type options: "bar" | "line" | "scatter" | "pie" | "area"

Required JSON format:
{{
  "chart_type": "bar",
  "title": "Chart title here",
  "x_label": "X axis label",
  "y_label": "Y axis label",
  "data": [
    {{
      "x": ["label1", "label2", "..."],
      "y": [value1, value2, "..."],
      "type": "bar",
      "name": "Series name"
    }}
  ],
  "layout": {{
    "showlegend": true,
    "colorway": ["#6c63ff", "#22d3a5", "#f59e0b"]
  }},
  "insight": "One sentence describing the most important finding in this chart."
}}

User request: "{query}"

Data context:
{context}
"""


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class DashboardEngine:

    async def generate(self, query: str, raw_data: str = "") -> dict:
        """
        Generate a Plotly chart spec from a user query.

        Args:
            query:    The user's request e.g. "show monthly revenue as a bar chart"
            raw_data: Optional CSV/JSON string the user pasted or uploaded directly.
                      If empty, we fall back to ChromaDB context.

        Returns:
            A dict with keys: chart_type, title, x_label, y_label,
            data, layout, insight, confidence.
            On failure: { "error": "...", "confidence": 0.0 }
        """
        # Step 1 — build context: prefer raw_data, fall back to RAG
        context = self._prepare_context(raw_data) or await rag.retrieve(query)

        if not context:
            context = "No data available. Generate a placeholder chart with sample structure."

        # Step 2 — ask the model
        prompt = DASHBOARD_PROMPT.format(query=query.strip(), context=context)
        raw    = await ollama_client.complete(prompt, max_tokens=1200)

        # Step 3 — parse and validate
        spec = self._parse(raw)
        if "error" in spec:
            return spec

        # Step 4 — score and return
        spec["confidence"] = self._score(spec)
        return spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_context(self, raw_data: str) -> str:
        """Clean and truncate raw CSV/JSON pasted by the user."""
        if not raw_data or not raw_data.strip():
            return ""
        # Truncate to 2000 chars so it fits in the context window comfortably
        truncated = raw_data.strip()[:2000]
        return f"User-provided data:\n{truncated}"

    def _parse(self, raw: str) -> dict:
        """
        Safely extract the JSON spec from the model's response.
        Handles markdown fences, leading text, trailing commentary.
        """
        # Remove markdown fences
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find the first complete {...} block
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {
            "error": "Model returned an unparseable response. Try rephrasing your request.",
            "raw":   raw[:300],
            "confidence": 0.0,
        }

    def _score(self, spec: dict) -> float:
        """
        Heuristic confidence score based on how complete the spec is.
        Returned to the frontend so it can show a quality indicator.

        Scoring breakdown:
          chart_type present      → 0.15
          title present           → 0.10
          data array non-empty    → 0.35
          x values present        → 0.20
          y values present        → 0.10
          insight present         → 0.10
        """
        score = 0.0

        if spec.get("chart_type"):
            score += 0.15
        if spec.get("title"):
            score += 0.10
        if spec.get("data") and len(spec["data"]) > 0:
            score += 0.35
            first = spec["data"][0]
            if first.get("x") and len(first["x"]) > 0:
                score += 0.20
            if first.get("y") and len(first["y"]) > 0:
                score += 0.10
        if spec.get("insight"):
            score += 0.10

        return round(score, 2)


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.dashboard_engine import dashboard_engine
# ------------------------------------------------------------------
dashboard_engine = DashboardEngine()
