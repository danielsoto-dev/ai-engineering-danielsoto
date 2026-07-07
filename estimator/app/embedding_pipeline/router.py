"""POST /embeddings/ingest (Session 7 pre-exercise, Step 5)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

_chunker = JSONStructuralChunker()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    chunks = _chunker.chunk(request.budgets)
    try:
        embedded = OpenAIEmbedder().embed_many(chunks)
    except Exception:
        log.exception("embedding_ingest_failed", chunk_count=len(chunks))
        raise HTTPException(status_code=500, detail="embedding generation failed") from None

    total_tokens = sum(c.token_count for c in chunks)
    return IngestResponse(
        chunks=embedded,
        stats=IngestStats(
            total_budgets=len(request.budgets),
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(total_tokens),
        ),
    )
