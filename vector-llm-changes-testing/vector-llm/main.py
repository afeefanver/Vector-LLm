"""
main.py
=======
The Vector LLM Microservice.
Runs on port 8001. Your team calls these endpoints — they own everything else.

ENDPOINTS:
  POST /query       — general question about data (auto-dispatches by intent)
  POST /dashboard   — generate a Plotly chart spec
  POST /decide      — get a decision recommendation
  POST /upload      — ingest a file into ChromaDB
  GET  /upload      — list this user's ingested documents
  DELETE /upload/{doc_id} — remove a document
  GET  /health      — check all services are running (no auth required)

AUTH:
  All endpoints except /health require:
      X-API-Key: <your_api_key>
  Keys are configured via API_KEYS env var (see auth.py).

START:
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
import os
import uuid

from auth import get_user_id
from llm.ollama import ollama_client
from llm.rag import rag
from llm.router import router as llm_router, Intent
from llm.decision_engine import decision_engine
from llm.credits import credits, InsufficientCreditsError
from config import settings


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run health checks on startup so issues surface immediately."""
    print("\n[vector-llm] starting up...")

    ollama_status = await ollama_client.health_check()
    chroma_status = await rag.health_check()

    if not ollama_status["ollama_running"]:
        print(f"[vector-llm] WARNING: Ollama not running. Fix: {ollama_status.get('fix')}")
    else:
        print(f"[vector-llm] Ollama OK — model: {ollama_status['active_model']}")

    if not chroma_status["chroma_running"]:
        print(f"[vector-llm] WARNING: ChromaDB not running. Fix: {chroma_status.get('fix')}")
    else:
        print(f"[vector-llm] ChromaDB OK")

    print("[vector-llm] ready on port 8001\n")
    yield

    # Shutdown -- drain the shared httpx client cleanly
    await ollama_client.close()
    await credits.close()
    print("[vector-llm] shutdown complete")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

# Allowed origins from env — defaults to localhost only (not "*")
_ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080"
).split(",")

app = FastAPI(
    title       = "Vector LLM Microservice",
    description = "Local LLM layer — dashboard generation and decision intelligence",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = _ALLOWED_ORIGINS,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:  str
    stream: bool = False    # set True for token-by-token streaming

class QueryResponse(BaseModel):
    answer:     str
    intent:     str
    confidence: float

class DashboardRequest(BaseModel):
    query:    str
    raw_data: str = ""      # optional CSV/JSON pasted by the user

class DecideRequest(BaseModel):
    query:    str
    csv_data: str = ""      # optional CSV for statistical analysis


# ------------------------------------------------------------------
# POST /query  — auto-dispatches to the right engine by intent
# ------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, summary="Ask a question about your data")
async def query(req: QueryRequest, user_id: str = Depends(get_user_id)):
    """
    General-purpose question answering with automatic intent dispatch.

    The router classifies the query intent, then dispatches to the right engine:
      - DASHBOARD  → dashboard_engine  (returns a Plotly spec as the answer)
      - DECISION   → decision_engine   (returns structured recommendation)
      - GENERAL    → ollama direct     (plain language answer with RAG context)

    Set stream=true for token-by-token streaming (GENERAL intent only).
    """
    route = await llm_router.route(req.query)

    # --- Intent dispatch ---
    # Previously the intent was detected but ignored — every query fell through
    # to ollama_client.complete(). Now each intent routes to the right engine.

    if route.intent == Intent.DASHBOARD:
        # Retrieve any previously uploaded data for this user as context
        raw_data = await rag.retrieve(req.query, user_id=user_id)
        spec     = await dashboard_engine.generate(
            query    = route.refined_query,
            raw_data = raw_data,
        )
        return QueryResponse(
            answer     = json.dumps(spec),
            intent     = route.intent.value,
            confidence = route.confidence,
        )

    if route.intent == Intent.DECISION:
        csv_context = await rag.retrieve(req.query, user_id=user_id)
        result      = await decision_engine.decide(
            query    = route.refined_query,
            csv_data = csv_context,
        )
        return QueryResponse(
            answer     = json.dumps(result),
            intent     = route.intent.value,
            confidence = result.get("confidence", route.confidence),
        )

    # GENERAL intent — plain language answer with RAG context injected
    context = await rag.retrieve(req.query, user_id=user_id)

    prompt = f"""{context}

User question: {route.refined_query}

Answer clearly and concisely based on the data context above.
If no context is available, say so and answer from general knowledge."""

    if req.stream:
        async def token_stream():
            async for token in ollama_client.stream(prompt):
                yield token
        return StreamingResponse(token_stream(), media_type="text/plain")

    answer = await ollama_client.complete(prompt)
    return QueryResponse(
        answer     = answer.strip(),
        intent     = route.intent.value,
        confidence = route.confidence,
    )


# ------------------------------------------------------------------
# POST /dashboard
# ------------------------------------------------------------------

@app.post("/dashboard", summary="Generate a Plotly chart spec")
async def dashboard(req: DashboardRequest, user_id: str = Depends(get_user_id)):

    cost = credits.cost_for("dashboard")

    try:
        await credits.check_and_reserve(user_id, cost)
    except InsufficientCreditsError as e:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Have {e.available}, need {e.required}."
        )

    spec = await dashboard_engine.generate(
        query=req.query,
        raw_data=req.raw_data,
        user_id=user_id,
    )

    await credits.deduct(
        user_id,
        cost,
        "dashboard",
        tokens_used=800,
    )

    return spec


# ------------------------------------------------------------------
# POST /decide
# ------------------------------------------------------------------

@app.post("/decide", summary="Get a data-driven decision recommendation")
async def decide(req: DecideRequest, user_id: str = Depends(get_user_id)):
    cost = credits.cost_for("decide")

    try:
        await credits.check_and_reserve(user_id, cost)
    except InsufficientCreditsError as e:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Have {e.available}, need {e.required}."
        )

    try:
        csv_data = req.csv_data
        if not csv_data:
            csv_data = await rag.retrieve(req.query, user_id=user_id)

        result = await decision_engine.decide(
            query=req.query,
            csv_data=csv_data,
        )

        await credits.deduct(
            user_id,
            cost,
            "decide",
            tokens_used=800,
        )

        return result

    except Exception:
        # only if your credits module supports it
        await credits.release(user_id, cost)
        raise


# ------------------------------------------------------------------
# POST /upload
# ------------------------------------------------------------------

SUPPORTED_TYPES = {".csv", ".txt", ".md", ".json"}

@app.post("/upload", summary="Ingest a file into your knowledge base")
async def upload(file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    """
    Upload a file and store it in this user's isolated ChromaDB collection.

    Supported formats: .csv  .txt  .json  .md

    After upload, all /query, /dashboard, and /decide calls for this user
    will automatically retrieve relevant context from this file.
    Other users cannot see or retrieve this file.
    """
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_TYPES)}",
        )

    content = await file.read()
    text    = content.decode("utf-8", errors="ignore").strip()

    if not text:
        raise HTTPException(status_code=400, detail="File is empty.")

    doc_id = str(uuid.uuid4())
    result = await rag.ingest(
        text     = text,
        user_id  = user_id,
        doc_id   = doc_id,
        metadata = {"filename": file.filename, "ext": ext},
    )

    return {
        "message":       f"{file.filename} ingested successfully.",
        "doc_id":        doc_id,
        "chunks_stored": result["chunks_stored"],
        "characters":    len(text),
    }


# ------------------------------------------------------------------
# GET /upload  — list this user's documents
# ------------------------------------------------------------------

@app.get("/upload", summary="List your ingested documents")
async def list_uploads(user_id: str = Depends(get_user_id)):
    """
    Returns all documents previously uploaded by this user.
    Use the doc_id values here to delete specific files via DELETE /upload/{doc_id}.
    """
    docs = await rag.list_docs(user_id=user_id)
    return {"documents": docs, "count": len(docs)}


# ------------------------------------------------------------------
# DELETE /upload/{doc_id}
# ------------------------------------------------------------------

@app.delete("/upload/{doc_id}", summary="Remove a file from your knowledge base")
async def delete_upload(doc_id: str, user_id: str = Depends(get_user_id)):
    """
    Remove all chunks for a previously uploaded document.
    Only removes from this user's collection — other users are unaffected.
    """
    result = await rag.delete_doc(doc_id=doc_id, user_id=user_id)
    return result


# ------------------------------------------------------------------
# GET /health  — no auth required (used by load balancers, uptime monitors)
# ------------------------------------------------------------------

@app.get("/health", summary="Check all services are running")
async def health():
    """
    Returns status of Ollama and ChromaDB.
    This endpoint does not require an API key.
    """
    ollama_status = await ollama_client.health_check()
    chroma_status = await rag.health_check()

    all_ok = ollama_status["ollama_running"] and chroma_status["chroma_running"]

    return {
        "status":   "ok" if all_ok else "degraded",
        "ollama":   ollama_status,
        "chromadb": chroma_status,
        "model":    settings.OLLAMA_MODEL,
    }
