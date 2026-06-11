"""
tests/test_api.py
=================
End-to-end tests that hit the live FastAPI endpoints.
Run the server first, then run this file.

HOW TO RUN:
  # Terminal 1
  ollama serve

  # Terminal 2
  chroma run --path ./chroma_data

  # Terminal 3
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload

  # Terminal 4
  python tests/test_api.py
"""

import asyncio
import sys
import os
import json
import httpx

BASE = "http://localhost:8001"
TIMEOUT = 300   # CPU needs time for LLM calls

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


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

async def test_health(client: httpx.AsyncClient):
    print("\n── Test 1: GET /health ──")
    resp = await client.get(f"{BASE}/health")
    data = resp.json()

    print(f"  status:          {data.get('status')}")
    print(f"  ollama running:  {data['ollama'].get('ollama_running')}")
    print(f"  chroma running:  {data['chromadb'].get('chroma_running')}")
    print(f"  model:           {data.get('model')}")

    assert resp.status_code == 200
    assert data["ollama"]["ollama_running"],  "Ollama not running"
    assert data["chromadb"]["chroma_running"], "ChromaDB not running"
    print("  ✓ All services healthy")


async def test_upload(client: httpx.AsyncClient) -> str:
    print("\n── Test 2: POST /upload ──")
    files  = {"file": ("sales.csv", REGIONAL_CSV.encode(), "text/csv")}
    resp   = await client.post(f"{BASE}/upload", files=files)
    data   = resp.json()

    print(f"  status code:    {resp.status_code}")
    print(f"  doc_id:         {data.get('doc_id')}")
    print(f"  chunks_stored:  {data.get('chunks_stored')}")
    print(f"  characters:     {data.get('characters')}")

    assert resp.status_code == 200
    assert data.get("doc_id"),            "No doc_id returned"
    assert data.get("chunks_stored", 0) > 0, "No chunks stored"
    print("  ✓ File uploaded and ingested")
    return data["doc_id"]


async def test_query(client: httpx.AsyncClient):
    print("\n── Test 3: POST /query ──")
    payload = {"query": "Which region had higher revenue?", "stream": False}
    resp    = await client.post(f"{BASE}/query", json=payload)
    data    = resp.json()

    print(f"  status code:  {resp.status_code}")
    print(f"  intent:       {data.get('intent')}")
    print(f"  confidence:   {data.get('confidence')}")
    print(f"  answer:       {data.get('answer', '')[:100]}")

    assert resp.status_code == 200
    assert data.get("answer"),     "No answer returned"
    assert data.get("intent"),     "No intent returned"
    assert data.get("confidence"), "No confidence returned"
    print("  ✓ Query answered")


async def test_dashboard(client: httpx.AsyncClient):
    print("\n── Test 4: POST /dashboard ──")
    payload = {
        "query":    "show monthly revenue as a bar chart",
        "raw_data": REGIONAL_CSV,
    }
    resp = await client.post(f"{BASE}/dashboard", json=payload)
    data = resp.json()

    print(f"  status code:  {resp.status_code}")
    print(f"  chart_type:   {data.get('chart_type')}")
    print(f"  title:        {data.get('title')}")
    print(f"  series:       {len(data.get('data', []))} series")
    print(f"  confidence:   {data.get('confidence')}")
    print(f"  insight:      {data.get('insight', '')[:80]}")

    assert resp.status_code == 200
    assert data.get("chart_type"), "No chart_type returned"
    assert data.get("data"),       "No data series returned"
    print("  ✓ Dashboard spec generated")


async def test_decide(client: httpx.AsyncClient):
    print("\n── Test 5: POST /decide ──")
    payload = {
        "query":    "Should we focus budget on North or South India?",
        "csv_data": REGIONAL_CSV,
    }
    resp = await client.post(f"{BASE}/decide", json=payload)
    data = resp.json()

    print(f"  status code:      {resp.status_code}")
    print(f"  recommendation:   {data.get('recommendation', '')[:90]}")
    print(f"  confidence:       {data.get('confidence')}")
    print(f"  risk:             {data.get('risk')}")
    print(f"  needs_more_data:  {data.get('needs_more_data')}")

    assert resp.status_code == 200
    assert data.get("recommendation"), "No recommendation returned"
    assert data.get("confidence"),     "No confidence returned"
    print("  ✓ Decision generated")


async def test_delete_upload(client: httpx.AsyncClient, doc_id: str):
    print(f"\n── Test 6: DELETE /upload/{doc_id} ──")
    resp = await client.delete(f"{BASE}/upload/{doc_id}")
    data = resp.json()

    print(f"  status code:     {resp.status_code}")
    print(f"  chunks_removed:  {data.get('chunks_removed')}")

    assert resp.status_code == 200
    print("  ✓ Document deleted")


async def test_unsupported_upload(client: httpx.AsyncClient):
    print("\n── Test 7: unsupported file type → 400 ──")
    files = {"file": ("report.pdf", b"fake pdf content", "application/pdf")}
    resp  = await client.post(f"{BASE}/upload", files=files)

    print(f"  status code:  {resp.status_code}")
    print(f"  detail:       {resp.json().get('detail')}")

    assert resp.status_code == 400
    print("  ✓ Unsupported file correctly rejected")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

async def main():
    print("=" * 52)
    print("  Vector LLM — end-to-end API tests")
    print("=" * 52)
    print(f"  Target: {BASE}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Check server is reachable first
        try:
            await client.get(f"{BASE}/health")
        except httpx.ConnectError:
            print(f"\n  ERROR: Cannot connect to {BASE}")
            print("  Make sure the server is running:")
            print("  uvicorn main:app --host 0.0.0.0 --port 8001 --reload")
            sys.exit(1)

        await test_health(client)
        doc_id = await test_upload(client)
        await test_query(client)
        await test_dashboard(client)
        await test_decide(client)
        await test_delete_upload(client, doc_id)
        await test_unsupported_upload(client)

    print("\n" + "=" * 52)
    print("  All API tests passed.")
    print("  Your microservice is ready to hand off to the team.")
    print(f"  Endpoint: {BASE}")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())
