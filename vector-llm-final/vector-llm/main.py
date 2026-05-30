"""
main.py
=======
The Vector LLM Microservice.
Runs on port 8001. Your team calls these endpoints — they own everything else.

ENDPOINTS:
  POST /query       — general question about data
  POST /dashboard   — generate a Plotly chart spec
  POST /decide      — get a decision recommendation
  POST /upload      — ingest a file into ChromaDB
  GET  /health      — check all services are running

START:
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uuid

from llm.ollama import ollama_client
from llm.rag import rag
from llm.router import router as llm_router, Intent
from llm.dashboard_engine import dashboard_engine
from llm.decision_engine import decision_engine
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
        print(f"[vector-llm] ChromaDB OK — chunks stored: {chroma_status['chunks_stored']}")

    print("[vector-llm] ready on port 8001\n")
    yield


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title       = "Vector LLM Microservice",
    description = "Local LLM layer — dashboard generation and decision intelligence",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten this in production to your team's domain
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:  str
    stream: bool = False        # set True for token-by-token streaming

class QueryResponse(BaseModel):
    answer:     str
    intent:     str
    confidence: float

class DashboardRequest(BaseModel):
    query:    str
    raw_data: str = ""          # optional CSV/JSON pasted by the user

class DecideRequest(BaseModel):
    query:    str
    csv_data: str = ""          # optional CSV for statistical analysis


# ------------------------------------------------------------------
# POST /query
# ------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, summary="Ask a question about your data")
async def query(req: QueryRequest):
    """
    General-purpose question answering.

    - Routes the query through the intent classifier first
    - Retrieves relevant context from ChromaDB (if files have been uploaded)
    - Returns the answer plus the detected intent and confidence

    Set stream=true to receive tokens progressively (text/plain response).
    """
    route   = await llm_router.route(req.query)
    context = await rag.retrieve(req.query)

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
async def dashboard(req: DashboardRequest):
    """
    Generate a Plotly-compatible chart spec from a user query and data.

    Accepts:
    - query:    What the user wants to visualise
    - raw_data: Optional CSV/JSON string pasted directly

    Returns a JSON object your React frontend can pass directly to Plotly.
    If raw_data is empty, uses ChromaDB context from previously uploaded files.
    """
    spec = await dashboard_engine.generate(
        query    = req.query,
        raw_data = req.raw_data,
    )
    return spec


# ------------------------------------------------------------------
# POST /decide
# ------------------------------------------------------------------

@app.post("/decide", summary="Get a data-driven decision recommendation")
async def decide(req: DecideRequest):
    """
    Analyse data and return a structured recommendation.

    How accuracy stays above 90%:
    1. pandas computes exact statistics from the CSV
    2. The LLM reads those pre-computed numbers — it never does maths itself
    3. ChromaDB provides supporting context from previously uploaded files

    Returns: recommendation, reasoning, confidence, risk, alternatives, key_metrics.
    If confidence < 0.75, needs_more_data is set to true.
    """
    result = await decision_engine.decide(
        query    = req.query,
        csv_data = req.csv_data,
    )
    return result


# ------------------------------------------------------------------
# POST /upload
# ------------------------------------------------------------------

SUPPORTED_TYPES = {".csv", ".txt", ".md", ".json"}

@app.post("/upload", summary="Ingest a file into the knowledge base")
async def upload(file: UploadFile = File(...)):
    """
    Upload a file and store it in ChromaDB for RAG retrieval.

    Supported formats: .csv  .txt  .json  .md
    After upload, all subsequent /query, /dashboard, and /decide calls
    will automatically retrieve relevant context from this file.
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
        doc_id   = doc_id,
        metadata = {"filename": file.filename, "ext": ext},
    )

    return {
        "message":        f"{file.filename} ingested successfully.",
        "doc_id":         doc_id,
        "chunks_stored":  result["chunks_stored"],
        "characters":     len(text),
    }


# ------------------------------------------------------------------
# DELETE /upload/{doc_id}
# ------------------------------------------------------------------

@app.delete("/upload/{doc_id}", summary="Remove a file from the knowledge base")
async def delete_upload(doc_id: str):
    """Remove all chunks for a previously uploaded document."""
    result = await rag.delete_doc(doc_id)
    return result


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------

@app.get("/health", summary="Check all services are running")
async def health():
    """
    Returns status of Ollama and ChromaDB.
    Your team can call this to verify the microservice is ready before use.
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
