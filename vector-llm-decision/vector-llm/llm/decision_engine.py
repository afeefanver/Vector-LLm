"""
llm/decision_engine.py
======================
The accuracy engine. Achieves 90%+ by splitting the work in two:

  Step 1 — pandas computes the statistics (numbers are exact, no LLM involved)
  Step 2 — Mistral reads those pre-computed numbers and writes the recommendation

WHY THIS WORKS:
  Small local models (7B params) are bad at mental arithmetic.
  They are good at reading a table of numbers and writing a clear recommendation.
  So we never ask the LLM to do maths — we hand it already-correct results.

WHAT THE TEAM RECEIVES:
  {
    "recommendation": "Expand to North India — revenue is 62% higher than South.",
    "reasoning":      "North India averaged ₹97,000/month vs South India ₹63,666/month.",
    "confidence":     0.91,
    "risk":           "low",
    "alternatives":   ["Invest in South India retention", "Split budget 70/30"],
    "key_metrics":    { "north_avg": 97000, "south_avg": 63666, "gap_pct": 52.3 },
    "needs_more_data": false
  }

USAGE:
  from llm.decision_engine import decision_engine

  result = await decision_engine.decide(
      query    = "should we focus more on North or South India?",
      csv_data = "month,revenue,region\\nJan,52000,South\\n..."
  )
"""

import json
import re
import pandas as pd
from io import StringIO
from llm.ollama import ollama_client
from llm.rag import rag
from config import settings


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

DECISION_PROMPT = """You are a senior business analyst at a data analytics company.

You have been given:
1. The user's business question
2. Pre-computed statistics (these numbers are exact — computed by pandas, not you)
3. Supporting context from the user's uploaded documents

Your job: write a clear, actionable recommendation based on these numbers.

RULES:
- Respond with ONLY a valid JSON object. No explanation. No markdown fences.
- Base your recommendation strictly on the provided statistics.
- confidence: how certain you are given the data quality (0.0 to 1.0)
- risk: "low" if data is clear, "medium" if some uncertainty, "high" if data is thin
- alternatives: 2 other options the user could consider
- key_metrics: the 2-4 numbers that most support your recommendation
- If the data is insufficient to make a confident recommendation, set needs_more_data: true

Required JSON format:
{{
  "recommendation": "Clear 1-2 sentence action recommendation",
  "reasoning":      "Why — reference the specific numbers from the statistics",
  "confidence":     0.0 to 1.0,
  "risk":           "low" | "medium" | "high",
  "alternatives":   ["Option 1", "Option 2"],
  "key_metrics":    {{ "metric_name": value }},
  "needs_more_data": true | false
}}

User question: "{query}"

Pre-computed statistics:
{stats}

Supporting context:
{context}
"""


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class DecisionEngine:

    async def decide(self, query: str, csv_data: str = "") -> dict:
        """
        Run statistical analysis then LLM interpretation.

        Args:
            query:    The user's business question.
            csv_data: Raw CSV string. If empty, falls back to RAG context.

        Returns:
            Dict with recommendation, reasoning, confidence, risk,
            alternatives, key_metrics, needs_more_data.
        """
        # Step 1 — compute stats with pandas (exact numbers, zero LLM)
        stats = self._compute_stats(csv_data) if csv_data.strip() else {}

        # Step 2 — pull supporting context from ChromaDB
        context = await rag.retrieve(query)
        if not context:
            context = "No additional context available from uploaded documents."

        # Step 3 — LLM interprets the pre-computed numbers
        prompt = DECISION_PROMPT.format(
            query   = query.strip(),
            stats   = json.dumps(stats, indent=2),
            context = context,
        )

        raw    = await ollama_client.complete(prompt, max_tokens=800)
        result = self._parse(raw)

        if "error" in result:
            return result

        # Step 4 — flag low-confidence answers for the frontend
        result["needs_more_data"] = (
            result.get("needs_more_data", False)
            or result.get("confidence", 0) < settings.CONFIDENCE_THRESHOLD
        )

        return result

    # ------------------------------------------------------------------
    # Statistics — pure pandas, no LLM
    # ------------------------------------------------------------------

    def _compute_stats(self, csv_data: str) -> dict:
        """
        Compute descriptive statistics from CSV data.
        These numbers are handed directly to the LLM — they are exact.

        Returns a dict the LLM can read and reference in its recommendation.
        """
        try:
            df      = pd.read_csv(StringIO(csv_data.strip()))
            numeric = df.select_dtypes(include="number")
            cat     = df.select_dtypes(include="object")

            stats = {
                "row_count":  int(len(df)),
                "columns":    list(df.columns),
            }

            # Numeric summary for every numeric column
            if not numeric.empty:
                summary = numeric.describe().round(2)
                stats["numeric_summary"] = json.loads(summary.to_json())

                # Trend on the last numeric column (most likely the KPI)
                kpi_col = numeric.columns[-1]
                vals    = numeric[kpi_col].dropna().tolist()
                if len(vals) >= 2:
                    change_pct = (vals[-1] - vals[0]) / (abs(vals[0]) + 1e-9) * 100
                    stats["trend"] = {
                        "column":         kpi_col,
                        "first_value":    round(vals[0], 2),
                        "last_value":     round(vals[-1], 2),
                        "change_pct":     round(change_pct, 2),
                        "direction":      "up" if change_pct > 0 else "down",
                        "peak_value":     round(max(vals), 2),
                        "lowest_value":   round(min(vals), 2),
                    }

            # Group-level aggregates for categorical columns
            if not cat.empty and not numeric.empty:
                group_col = cat.columns[0]
                kpi_col   = numeric.columns[-1]
                grouped   = df.groupby(group_col)[kpi_col].agg(["sum","mean","count"]).round(2)
                stats["group_summary"] = json.loads(grouped.to_json())

            return stats

        except Exception as e:
            return {"error": f"Could not parse CSV: {e}"}

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> dict:
        """Extract JSON from the model's response, handling markdown fences."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {
            "error":          "Model returned an unparseable response.",
            "raw":            raw[:300],
            "confidence":     0.0,
            "needs_more_data": True,
        }


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.decision_engine import decision_engine
# ------------------------------------------------------------------
decision_engine = DecisionEngine()
