"""
tests/test_dashboard_engine.py
==============================
Run this to verify dashboard_engine.py is working.

HOW TO RUN:
  ollama serve                       ← terminal 1
  chroma run --path ./chroma_data    ← terminal 2
  python tests/test_dashboard_engine.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.dashboard_engine import dashboard_engine
from llm.rag import rag


SAMPLE_CSV = """month,revenue,units_sold
January,52000,430
February,61000,510
March,48000,390
April,73000,620
May,81000,700
June,67000,560
July,90000,780
August,95000,820
September,88000,760
October,102000,890
November,115000,980
December,130000,1100
"""


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

async def test_generate_with_raw_data():
    """Most common case — user pastes CSV and asks for a chart."""
    print("\n── Test 1: generate chart from raw CSV data ──")

    spec = await dashboard_engine.generate(
        query    = "show monthly revenue as a bar chart",
        raw_data = SAMPLE_CSV,
    )

    print(f"  chart_type:  {spec.get('chart_type')}")
    print(f"  title:       {spec.get('title')}")
    print(f"  series:      {len(spec.get('data', []))} series")
    print(f"  x values:    {spec.get('data', [{}])[0].get('x', [])[:4]}...")
    print(f"  y values:    {spec.get('data', [{}])[0].get('y', [])[:4]}...")
    print(f"  insight:     {spec.get('insight', '')[:80]}")
    print(f"  confidence:  {spec.get('confidence')}")

    assert "error"      not in spec,        f"Got error: {spec.get('error')}"
    assert spec.get("chart_type"),          "Missing chart_type"
    assert spec.get("data"),                "Missing data array"
    assert spec["confidence"] >= 0.5,       f"Low confidence: {spec['confidence']}"
    print("  ✓ Chart spec generated from raw data")


async def test_generate_with_rag():
    """User asks without pasting data — RAG retrieves context from ChromaDB."""
    print("\n── Test 2: generate chart using RAG context ──")

    # Ingest first so there's something to retrieve
    await rag.ingest(
        text     = SAMPLE_CSV,
        doc_id   = "dash_test_csv",
        metadata = {"filename": "sales.csv"},
    )

    spec = await dashboard_engine.generate(
        query    = "plot units sold over the year as a line chart",
        raw_data = "",   # no raw data — must come from RAG
    )

    print(f"  chart_type:  {spec.get('chart_type')}")
    print(f"  title:       {spec.get('title')}")
    print(f"  confidence:  {spec.get('confidence')}")

    assert "error" not in spec, f"Got error: {spec.get('error')}"
    print("  ✓ Chart spec generated via RAG context")

    # Clean up
    await rag.delete_doc("dash_test_csv")


async def test_generate_pie_chart():
    """Test a different chart type to ensure the model picks appropriately."""
    print("\n── Test 3: pie chart request ──")

    region_data = """region,revenue
North India,580000
South India,320000
East India,210000
West India,290000
"""

    spec = await dashboard_engine.generate(
        query    = "show revenue breakdown by region as a pie chart",
        raw_data = region_data,
    )

    print(f"  chart_type:  {spec.get('chart_type')}")
    print(f"  title:       {spec.get('title')}")
    print(f"  confidence:  {spec.get('confidence')}")

    assert "error" not in spec, f"Got error: {spec.get('error')}"
    print("  ✓ Pie chart spec generated")


async def test_generate_no_data():
    """No data provided and ChromaDB is empty — should return gracefully."""
    print("\n── Test 4: generate with no data at all ──")

    spec = await dashboard_engine.generate(
        query    = "show me something useful",
        raw_data = "",
    )

    # Should not crash — returns a spec (possibly placeholder) or error dict
    print(f"  Result keys: {list(spec.keys())}")
    print(f"  confidence:  {spec.get('confidence', 0)}")
    assert isinstance(spec, dict), "Must always return a dict"
    print("  ✓ Returns a dict gracefully even with no data")


async def test_confidence_scoring():
    """Verify confidence score reflects spec completeness."""
    print("\n── Test 5: confidence scoring ──")

    # Full spec → high score
    full_spec = {
        "chart_type": "bar",
        "title":      "Test",
        "data":       [{"x": ["a", "b"], "y": [1, 2], "type": "bar", "name": "s1"}],
        "insight":    "Some insight here.",
    }
    score_full = dashboard_engine._score(full_spec)

    # Minimal spec → low score
    minimal_spec = {"chart_type": "bar"}
    score_minimal = dashboard_engine._score(minimal_spec)

    print(f"  Full spec score:    {score_full}  (expect ≥ 0.9)")
    print(f"  Minimal spec score: {score_minimal}  (expect ≤ 0.2)")

    assert score_full    >= 0.9, f"Full spec score too low: {score_full}"
    assert score_minimal <= 0.2, f"Minimal spec score too high: {score_minimal}"
    print("  ✓ Confidence scoring behaves correctly")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

async def main():
    print("=" * 52)
    print("  Vector LLM — dashboard_engine.py tests")
    print("=" * 52)

    await test_generate_with_raw_data()
    await test_generate_with_rag()
    await test_generate_pie_chart()
    await test_generate_no_data()
    await test_confidence_scoring()

    print("\n" + "=" * 52)
    print("  All dashboard tests done.")
    print("  Next: decision_engine.py")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())
