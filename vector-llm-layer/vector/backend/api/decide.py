"""api/decide.py"""
from fastapi import APIRouter
from pydantic import BaseModel
from llm.decision_engine import engine

router = APIRouter()

class DecisionRequest(BaseModel):
    query: str
    csv_data: str = ""

@router.post("")
async def decide(req: DecisionRequest):
    result = await engine.decide(req.query, req.csv_data)
    return result
