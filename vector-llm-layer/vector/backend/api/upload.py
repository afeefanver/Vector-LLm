"""api/upload.py — Ingest uploaded files into ChromaDB"""
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from llm.rag import retriever

router = APIRouter()

SUPPORTED = {".csv", ".txt", ".md", ".json"}

@router.post("")
async def upload(file: UploadFile = File(...)):
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type. Supported: {SUPPORTED}")

    content = await file.read()
    text    = content.decode("utf-8", errors="ignore")
    doc_id  = str(uuid.uuid4())

    await retriever.ingest(
        text     = text,
        doc_id   = doc_id,
        metadata = {"filename": file.filename},
    )

    return {
        "message":   f"{file.filename} ingested successfully",
        "doc_id":    doc_id,
        "characters": len(text),
    }
