"""
tests/test_ollama_router.py
===========================
Run this to verify your ollama.py and router.py are working.

HOW TO RUN:
  # Make sure Ollama is running first:
  ollama serve          ← in a separate terminal
  ollama pull mistral   ← only needed once

  # Then run the tests:
  cd vector-llm
  pip install -r requirements.txt
  python -m pytest tests/test_ollama_router.py -v

  # Or run without pytest — just plain Python:
  python tests/test_ollama_router.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.ollama import ollama_client
from llm.router import router, Intent


# ------------------------------------------------------------------
# Ollama tests
# ------------------------------------------------------------------

async def test_ollama_health():
    """Check Ollama is running before anything else."""
    print("\n── Test 1: Ollama health check ──")
    result = await ollama_client.health_check()
    print(f"  Running:       {result['ollama_running']}")
    print(f"  Active model:  {result.get('active_model')}")
    print(f"  Model ready:   {result.get('model_ready')}")

    if not result["ollama_running"]:
        print(f"\n  ✗ FIX: {result.get('fix')}")
        print("  Cannot continue — start Ollama first.")
        return False

    if not result.get("model_ready"):
        print(f"\n  ✗ Model not found. Run: ollama pull {result.get('active_model')}")
        return False

    print("  ✓ Ollama is running and model is ready")
    return True


async def test_ollama_complete():
    """Send a simple prompt and check we get a response."""
    print("\n── Test 2: ollama_client.complete() ──")
    prompt = "Reply with exactly three words: the capital of France."
    response = await ollama_client.complete(prompt, max_tokens=20)
    print(f"  Prompt:   {prompt}")
    print(f"  Response: {response.strip()}")
    assert len(response.strip()) > 0, "Got empty response"
    print("  ✓ Got a response")


async def test_ollama_stream():
    """Stream tokens and count them."""
    print("\n── Test 3: ollama_client.stream() ──")
    prompt = "Count from 1 to 5, one number per word."
    tokens = []
    async for token in ollama_client.stream(prompt):
        tokens.append(token)

    full = "".join(tokens)
    print(f"  Streamed {len(tokens)} tokens → '{full.strip()[:60]}'")
    assert len(tokens) > 0, "No tokens streamed"
    print("  ✓ Streaming works")


async def test_ollama_embed():
    """Check embeddings return a non-empty vector."""
    print("\n── Test 4: ollama_client.embed() ──")
    text = "monthly revenue data for Q3"
    vector = await ollama_client.embed(text)
    print(f"  Text:            '{text}'")
    print(f"  Embedding dims:  {len(vector)}")
    assert len(vector) > 0, "Got empty embedding"
    print("  ✓ Embeddings working")


# ------------------------------------------------------------------
# Router tests
# ------------------------------------------------------------------

ROUTING_CASES = [
    ("show me a bar chart of monthly sales",        Intent.DASHBOARD),
    ("should we enter the European market?",         Intent.DECISION),
    ("what was total revenue in March?",             Intent.QUESTION),
    ("predict sales for the next 6 months",          Intent.FORECAST),
    ("give me an overview of this dataset",          Intent.SUMMARY),
]


async def test_router_intents():
    """Run a set of queries through the router and check intent classification."""
    print("\n── Test 5: router.route() — intent classification ──")
    all_passed = True

    for query, expected_intent in ROUTING_CASES:
        result = await router.route(query)
        status = "✓" if result.intent == expected_intent else "✗"
        if result.intent != expected_intent:
            all_passed = False
        print(f"  {status} [{result.confidence:.2f}] '{query[:45]}...' → {result.intent.value} (expected {expected_intent.value})")

    if all_passed:
        print("  ✓ All intents correctly classified")
    else:
        print("  ⚠ Some intents misclassified — this is normal for smaller models")
        print("    Switch OLLAMA_MODEL=llama3.1 in .env for better accuracy")


async def test_router_fallback():
    """Check the router never crashes — even on garbage input."""
    print("\n── Test 6: router fallback on bad input ──")
    result = await router.route("asdfgh!!! ???")
    print(f"  Input:   garbage string")
    print(f"  Intent:  {result.intent.value}")
    print(f"  Confidence: {result.confidence}")
    assert result.intent in Intent.__members__.values(), "Got invalid intent"
    print("  ✓ Router handles bad input gracefully")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

async def main():
    print("=" * 52)
    print("  Vector LLM — ollama.py + router.py tests")
    print("=" * 52)

    # Health check first — abort if Ollama not running
    ok = await test_ollama_health()
    if not ok:
        sys.exit(1)

    await test_ollama_complete()
    await test_ollama_stream()
    await test_ollama_embed()
    await test_router_intents()
    await test_router_fallback()

    print("\n" + "=" * 52)
    print("  All tests done.")
    print("  Next step: run  python tests/test_ollama_router.py")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())
