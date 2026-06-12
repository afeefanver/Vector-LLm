"""db/database.py — Async PostgreSQL with SQLAlchemy"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Float, Text, DateTime, func
from core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class LLMLog(Base):
    __tablename__ = "llm_logs"
    id              = Column(String, primary_key=True)
    query           = Column(Text)
    intent          = Column(String)
    response        = Column(Text)
    confidence      = Column(Float)
    model_used      = Column(String)
    created_at      = Column(DateTime, server_default=func.now())

class Decision(Base):
    __tablename__ = "decisions"
    id              = Column(String, primary_key=True)
    query           = Column(Text)
    recommendation  = Column(Text)
    confidence      = Column(Float)
    risk            = Column(String)
    created_at      = Column(DateTime, server_default=func.now())

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
