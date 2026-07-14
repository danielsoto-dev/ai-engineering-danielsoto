"""Async engine/session factory for the pgvector persistence layer."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    # NullPool: a fresh connection per checkout, no pooling. Avoids handing
    # an asyncpg connection across event loops (e.g. TestClient's worker
    # thread vs pytest-asyncio's loop), which corrupts its protocol state.
    return create_async_engine(settings.DATABASE_URL, poolclass=NullPool)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a request-scoped async session."""
    async with get_sessionmaker()() as session:
        yield session
