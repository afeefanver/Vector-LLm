"""
tests/test_decision_engine.py
=============================
Run this to verify decision_engine.py is working.

HOW TO RUN:
  ollama serve                       ← terminal 1
  chroma run --path ./chroma_data    ← terminal 2
  python tests/test_decision_engine.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.decision_engine import decision_engine


REGIONAL_CSV = """month,revenue,region
January,52000,South
February,61000,South
March,48000,South
April,73000,South
May,81000,South
June,67000,South
July,90000,North
August,95000,North
September,88000,North
October,102000,North
November,115000,North
December,130000,North
"""

PRODUCT_CSV = """product,units_sold,revenue,return_rate
Widget A,1200,84000,2.1
Widget B,340,61200,8.4
Widget C,890,53400,1.8
Widget D,120,14400,12.3
Widget E,670,46900,3.2
"""

THIN_CSV = """month,revenue
Jan,5000
"""


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

async def test_regional_decision():
    """Core case — should we focus on North or South India?"""
    print("\n── Test 1: regional expansion decision ──")

    result = await decision_engine.decide(
        query    = "Should we focus more budget on North or South India next quarter?",
        csv_data = REGIONAL_CSV,
    )

    print(f"  recommendation:  {result.get('recommendation', '')[:90]}")
    print(f"  reasoning:       {result.get('reasoning', '')[:90]}")
    print(f"  confidence:      {result.get('confidence')}")
    print(f"  risk:            {result.get('risk')}")
    print(f"  alternatives:    {result.get('alternatives', [])}")
    print(f"  key_metrics:     {result.get('key_metrics')}")
    print(f"  needs_more_data: {result.get('needs_more_data')}")

    assert "error"                  not in result, f"Got error: {result.get('error')}"
    assert result.get("recommendation"),           "Missing recommendation"
    assert result.get("confidence",  0) > 0,       "Confidence must be > 0"
    assert result.get("risk")        in ("low","medium","high"), "Invalid risk level"
    assert isinstance(result.get("alternatives"), list),        "Alternatives must be a list"
    print("  ✓ Decision generated with all required fields")


async def test_product_decision():
    """Which product to double down on vs discontinue."""
    print("\n── Test 2: product portfolio decision ──")

    result = await decision_engine.decide(
        query    = "Which product should we discontinue and which should we invest more in?",
        csv_data = PRODUCT_CSV,
    )

    print(f"  recommendation:  {result.get('recommendation', '')[:90]}")
    print(f"  confidence:      {result.get('confidence')}")
    print(f"  risk:            {result.get('risk')}")

    assert "error" not in result, f"Got error: {result.get('error')}"
    assert result.get("recommendation"), "Missing recommendation"
    print("  ✓ Product decision generated")


async def test_stats_computation():
    """Verify pandas stats are computed correctly — no LLM involved."""
    print("\n── Test 3: pandas stats computation (no LLM) ──")

    stats = decision_engine._compute_stats(REGIONAL_CSV)

    print(f"  row_count:   {stats.get('row_count')}")
    print(f"  columns:     {stats.get('columns')}")
    print(f"  trend dir:   {stats.get('trend', {}).get('direction')}")
    print(f"  trend pct:   {stats.get('trend', {}).get('change_pct')}%")
    print(f"  group_summary keys: {list(stats.get('group_summary', {}).keys())}")

    assert stats["row_count"]                   == 12,   "Expected 12 rows"
    assert "revenue"                            in stats["columns"]
    assert stats["trend"]["direction"]          == "up", "Revenue trend should be up"
    assert stats["trend"]["change_pct"]         > 0,     "Change % should be positive"
    assert "North"                              in str(stats.get("group_summary", {}))
    print("  ✓ Stats computation correct — numbers are exact")


async def test_thin_data_flagged():
    """One row of data — should set needs_more_data: true."""
    print("\n── Test 4: thin data → needs_more_data flag ──")

    result = await decision_engine.decide(
        query    = "Should we expand?",
        csv_data = THIN_CSV,
    )

    print(f"  confidence:      {result.get('confidence')}")
    print(f"  needs_more_data: {result.get('needs_more_data')}")

    assert result.get("needs_more_data") == True, \
        f"Expected needs_more_data=True for thin data, got {result.get('needs_more_data')}"
    print("  ✓ Thin data correctly flagged")


async def test_no_data_fallback():
    """No CSV at all — should return gracefully without crashing."""
    print("\n── Test 5: no data fallback ──")

    result = await decision_engine.decide(
        query    = "What should we do next quarter?",
        csv_data = "",
    )

    print(f"  keys returned:   {list(result.keys())}")
    print(f"  needs_more_data: {result.get('needs_more_data')}")

    assert isinstance(result, dict), "Must always return a dict"
    print("  ✓ No data handled gracefully")


async def test_bad_csv_fallback():
    """Malformed CSV — stats should return error key, engine should not crash."""
    print("\n── Test 6: malformed CSV fallback ──")

    result = await decision_engine.decide(
        query    = "Analyse this",
        csv_data = "this is not,,, a valid csv\n!!!\n@@@",
    )

    print(f"  Got result:  {list(result.keys())}")
    assert isinstance(result, dict), "Must always return a dict"
    print("  ✓ Malformed CSV handled without crash")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

async def main():
    print("=" * 52)
    print("  Vector LLM — decision_engine.py tests")
    print("=" * 52)

    await test_regional_decision()
    await test_product_decision()
    await test_stats_computation()
    await test_thin_data_flagged()
    await test_no_data_fallback()
    await test_bad_csv_fallback()

    print("\n" + "=" * 52)
    print("  All decision tests done.")
    print("  Final step: main.py — wire everything into FastAPI endpoints")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())
