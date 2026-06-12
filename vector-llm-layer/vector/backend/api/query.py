"""api/query.py — General query endpoint with streaming support"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llm.router import router as llm_router, Intent
from llm.ollama_client import OllamaClient
from llm.rag import retriever

router = APIRouter()
client = OllamaClient()


class QueryRequest(BaseModel):
    query: str
    stream: bool = False


@router.post("")
async def query(req: QueryRequest):
    route   = await llm_router.route(req.query)
    context = await retriever.retrieve(req.query)

    prompt = f"""{context}

User question: {route.refined_query}

Answer clearly and concisely based on the data context above."""

    if req.stream:
        async def gen():
            async for token in client.stream(prompt):
                yield token
        return StreamingResponse(gen(), media_type="text/plain")

    response = await client.complete(prompt)
    return {
        "answer":   response,
        "intent":   route.intent,
        "confidence": route.confidence,
    }
