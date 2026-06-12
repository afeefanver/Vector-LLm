"""api/dashboard.py"""
from fastapi import APIRouter
from pydantic import BaseModel
from llm.dashboard_engine import engine

router = APIRouter()

class DashboardRequest(BaseModel):
    query: str
    raw_data: str = ""   # CSV or JSON string

@router.post("")
async def generate_dashboard(req: DashboardRequest):
    spec = await engine.generate(req.query, req.raw_data)
    return spec
