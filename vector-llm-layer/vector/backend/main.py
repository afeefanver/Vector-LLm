from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api import query, dashboard, decide, upload
from db.database import init_db
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Vector LLM API",
    description="Local LLM-powered dashboard and decision engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,     prefix="/api/query",     tags=["Query"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(decide.router,    prefix="/api/decide",    tags=["Decisions"])
app.include_router(upload.router,    prefix="/api/upload",    tags=["Upload"])

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.OLLAMA_MODEL}
