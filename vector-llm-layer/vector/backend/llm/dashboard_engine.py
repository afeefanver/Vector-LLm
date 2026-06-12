"""
llm/dashboard_engine.py
-----------------------
Takes a user request + data context and returns a Plotly chart spec.
The LLM decides: chart type, axes, title, color scheme.
The frontend renders it directly from the JSON spec — no extra work.
"""

import json
from llm.ollama_client import OllamaClient
from llm.rag import retriever
from core.config import settings


DASHBOARD_PROMPT = """You are a data visualisation expert.

Given the user request and data context below, generate a Plotly.js chart configuration.
Respond with ONLY a valid JSON object — no explanation, no markdown fences.

The JSON must have this structure:
{{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "area" | "heatmap",
  "title": "Chart title",
  "x_label": "X axis label",
  "y_label": "Y axis label",
  "data": [
    {{
      "x": [list of x values],
      "y": [list of y values],
      "name": "Series name",
      "type": "bar" | "scatter" | "pie"
    }}
  ],
  "layout": {{
    "showlegend": true | false,
    "colorway": ["#hex1", "#hex2"]
  }},
  "insight": "One sentence explaining what this chart shows"
}}

User request: "{query}"

Data context:
{context}
"""


class DashboardEngine:
    def __init__(self):
        self.client = OllamaClient()

    async def generate(self, query: str, raw_data: str = "") -> dict:
        """
        Generate a Plotly chart spec from a user query.
        raw_data: CSV string or JSON string the user uploaded.
        """
        # Pull relevant context from ChromaDB
        rag_context = await retriever.retrieve(query)
        context = raw_data or rag_context or "No data provided."

        prompt = DASHBOARD_PROMPT.format(query=query, context=context)
        raw = await self.client.complete(prompt, max_tokens=1500)

        try:
            # Strip any accidental markdown fences
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            spec  = json.loads(clean)
            spec["confidence"] = self._score(spec)
            return spec
        except json.JSONDecodeError:
            return {
                "error": "Could not parse chart spec",
                "raw": raw,
                "confidence": 0.0,
            }

    def _score(self, spec: dict) -> float:
        """Heuristic confidence score based on spec completeness."""
        score = 0.0
        if spec.get("chart_type"):              score += 0.2
        if spec.get("title"):                   score += 0.1
        if spec.get("data") and len(spec["data"]) > 0: score += 0.4
        if spec.get("data", [{}])[0].get("x"):  score += 0.15
        if spec.get("insight"):                 score += 0.15
        return round(score, 2)


engine = DashboardEngine()
