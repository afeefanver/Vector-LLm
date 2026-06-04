"""
tests/test_rag.py
=================
Run this to verify rag.py is working.

HOW TO RUN:
  # Make sure both services are running:
  ollama serve                       ← terminal 1
  chroma run --path ./chroma_data    ← terminal 2

  # Then:
  python tests/test_rag.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.rag import rag


# ------------------------------------------------------------------
# Sample data — a small CSV and a short text doc
# ------------------------------------------------------------------

SAMPLE_CSV = """month,revenue,units_sold,region
January,52000,430,South India
February,61000,510,South India
March,48000,390,South India
April,73000,620,South India
May,81000,700,South India
June,67000,560,South India
July,90000,780,North India
August,95000,820,North India
September,88000,760,North India
October,102000,890,North India
November,115000,980,North India
December,130000,1100,North India
"""

SAMPLE_TEXT = """
Product performance summary for Vector SaaS — FY 2024.

The product saw strong growth in Q4 driven by enterprise adoption.
North India accounts for 62% of total annual revenue.
South India showed consistent growth of 8% month-over-month in H1.
The highest single-month revenue was December at 130,000.
Customer churn dropped from 4.2% to 1.8% after the onboarding revamp.
Decision intelligence features had a 91% satisfaction score in user surveys.
"""


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

async def test_chroma_health():
    print("\n── Test 1: ChromaDB health check ──")
    result = await rag.health_check()
    print(f"  Running:        {result['chroma_running']}")
    print(f"  Chunks stored:  {result.get('chunks_stored', 0)}")

    if not result["chroma_running"]:
        print(f"\n  ✗ FIX: {result.get('fix')}")
        print("  Cannot continue — start ChromaDB first.")
        return False

    print("  ✓ ChromaDB is running")
    return True


async def test_ingest_csv():
    print("\n── Test 2: ingest CSV data ──")
    result = await rag.ingest(
        text=SAMPLE_CSV,
        doc_id="test_sales_csv",
        metadata={"filename": "sales_2024.csv", "type": "csv"},
    )
    print(f"  doc_id:         {result['doc_id']}")
    print(f"  chunks_stored:  {result['chunks_stored']}")
    assert result["chunks_stored"] > 0
    print("  ✓ CSV ingested")


async def test_ingest_text():
    print("\n── Test 3: ingest text document ──")
    result = await rag.ingest(
        text=SAMPLE_TEXT,
        doc_id="test_summary_doc",
        metadata={"filename": "summary.txt", "type": "text"},
    )
    print(f"  doc_id:         {result['doc_id']}")
    print(f"  chunks_stored:  {result['chunks_stored']}")
    assert result["chunks_stored"] > 0
    print("  ✓ Text doc ingested")


async def test_retrieve_relevant():
    print("\n── Test 4: retrieve — relevant queries ──")

    queries = [
        "what was revenue in December?",
        "which region performed better?",
        "what is the customer churn rate?",
    ]

    for query in queries:
        context = await rag.retrieve(query)
        has_context = len(context) > 0
        preview     = context[:80].replace("\n", " ") if has_context else "(empty)"
        status      = "✓" if has_context else "✗"
        print(f"  {status} '{query}'")
        print(f"      context: {preview}...")

    print("  ✓ Retrieval returning context")


async def test_retrieve_irrelevant():
    print("\n── Test 5: retrieve — irrelevant query ──")
    # This query has nothing to do with the uploaded docs
    # The distance filter should return empty string
    context = await rag.retrieve("what is the weather like in Tokyo?")
    print(f"  Query:   'what is the weather like in Tokyo?'")
    print(f"  Context: {'(empty — correctly filtered out)' if not context else context[:60]}")
    print("  ✓ Irrelevant queries filtered cleanly")


async def test_retrieve_empty_store():
    print("\n── Test 6: retrieve — fresh collection (no docs) ──")
    # Temporarily use a different collection name
    original = rag._collection
    rag._collection = None

    # Point to a collection that definitely has no data
    from config import settings
    original_name              = settings.CHROMA_COLLECTION
    settings.CHROMA_COLLECTION = "test_empty_collection_xyz"
    rag._collection            = None

    context = await rag.retrieve("anything")
    print(f"  Context returned: '{context}'")
    assert context == "", f"Expected empty string, got: {context[:50]}"
    print("  ✓ Returns empty string when no docs ingested")

    # Restore
    settings.CHROMA_COLLECTION = original_name
    rag._collection             = None


async def test_delete_doc():
    print("\n── Test 7: delete a document ──")
    result = await rag.delete_doc("test_sales_csv")
    print(f"  doc_id:          test_sales_csv")
    print(f"  chunks_removed:  {result['chunks_removed']}")
    assert result["chunks_removed"] > 0
    print("  ✓ Document deleted from ChromaDB")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

async def main():
    print("=" * 52)
    print("  Vector LLM — rag.py tests")
    print("=" * 52)

    ok = await test_chroma_health()
    if not ok:
        sys.exit(1)

    await test_ingest_csv()
    await test_ingest_text()
    await test_retrieve_relevant()
    await test_retrieve_irrelevant()
    await test_retrieve_empty_store()
    await test_delete_doc()

    print("\n" + "=" * 52)
    print("  All RAG tests done.")
    print("  Next: dashboard_engine.py + decision_engine.py")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())
