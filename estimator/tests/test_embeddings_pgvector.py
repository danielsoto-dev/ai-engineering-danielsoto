"""Integration tests for POST /embeddings/ingest and POST /search.

Runs against the real compose Postgres (pgvector's Vector type and
cosine_distance operator have no SQLite equivalent, so this is not mocked at
the DB layer). Only the OpenAI embedder is faked, to keep the suite offline
and free. Skips automatically if Postgres isn't reachable.
"""

from __future__ import annotations

import os

import pytest
import sqlalchemy
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.session import get_db_session
from app.embedding_pipeline import router as embeddings_router_module
from app.main import app

SAMPLE_BUDGET = {
    "budget_id": "TEST-001",
    "client_metadata": {"name": "Acme", "sector": "finance", "country": "ES"},
    "project_summary": "Test project for OAuth backend",
    "main_technology": "python",
    "year": 2024,
    "total_estimated_hours": 40,
    "components": [
        {
            "component_id": "C1",
            "name": "OAuth backend",
            "description": "OAuth 2.0 authentication backend",
            "tech_stack": ["python", "fastapi"],
            "estimated_hours": 40,
            "complexity": "medium",
            "dependencies": [],
        }
    ],
}


class FakeEmbedder:
    """Deterministic fake: same text -> same vector, based on its hash."""

    def embed_one(self, text: str) -> list[float]:
        seed = hash(text) % 1000 / 1000
        return [seed] * 1536

    def embed_many(self, chunks):
        from app.embedding_pipeline.schemas import EmbeddedChunk

        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=self.embed_one(chunk.text))
            for chunk in chunks
        ]


@pytest.fixture
async def clean_db():
    """Truncate on a private engine scoped to this fixture's own event loop.

    The schema itself is expected to already exist via ``alembic upgrade
    head`` — this only resets data between tests. A dedicated engine (rather
    than the app's cached ``get_engine()``) avoids leaking asyncpg
    connections across the different event loops that pytest-asyncio and
    ``TestClient`` each run.
    """
    # TRUNCATE destroys whatever database it points at, so this fixture refuses
    # to run against the dev corpus unless TEST_DATABASE_URL names a throwaway
    # one explicitly. Re-ingesting after an accidental wipe costs real embedding
    # calls, which is not something a test run should ever trigger.
    test_url = os.environ.get("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip(
            "set TEST_DATABASE_URL to a disposable database to run the pgvector "
            "tests; they TRUNCATE documents and chunks"
        )

    engine = create_async_engine(test_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(
                sqlalchemy.text("TRUNCATE documents, chunks RESTART IDENTITY CASCADE")
            )
            await conn.commit()
    except Exception:
        pytest.skip(
            "Postgres not reachable (docker compose up -d postgres && alembic upgrade head)"
        )
    finally:
        await engine.dispose()

    yield


@pytest.fixture
def client(clean_db, monkeypatch):
    monkeypatch.setattr(embeddings_router_module, "OpenAIEmbedder", FakeEmbedder)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db_session, None)


def _ingest_payload(source_path: str = "test.json#TEST-001") -> dict:
    return {
        "source_path": source_path,
        "document_type": "historical_budget",
        "content": SAMPLE_BUDGET,
    }


def test_ingest_persists_document_and_chunks(client: TestClient) -> None:
    response = client.post("/embeddings/ingest", json=_ingest_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_created"] == 1
    assert body["embedding_dimension"] == 1536
    assert body["document_id"] > 0


def test_ingest_duplicate_source_path_returns_409(client: TestClient) -> None:
    client.post("/embeddings/ingest", json=_ingest_payload())
    response = client.post("/embeddings/ingest", json=_ingest_payload())
    assert response.status_code == 409
    assert response.json()["detail"]["detail"] == "Document already ingested"


def test_search_returns_ranked_results(client: TestClient) -> None:
    client.post("/embeddings/ingest", json=_ingest_payload())
    response = client.post("/search", json={"query": "OAuth authentication", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "OAuth authentication"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["chunk_type"] == "budget_component"
    assert "OAuth" in result["content"]
    assert isinstance(result["distance"], float)
