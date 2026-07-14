"""POST /embeddings/ingest and POST /search."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk as ChunkRow
from app.db.models import Document
from app.db.session import get_db_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import (
    IngestDocumentRequest,
    IngestDocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

log = structlog.get_logger()

router = APIRouter(tags=["embeddings"])

_chunker = JSONStructuralChunker()

CHUNK_TYPE_BUDGET_COMPONENT = "budget_component"


@router.post("/embeddings/ingest", response_model=IngestDocumentResponse)
async def ingest(
    request: IngestDocumentRequest, session: AsyncSession = Depends(get_db_session)
) -> IngestDocumentResponse:
    t0 = time.perf_counter()

    existing = await session.scalar(
        select(Document).where(Document.source_path == request.source_path)
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={"detail": "Document already ingested", "document_id": existing.id},
        )

    chunks = _chunker.chunk([request.content])
    try:
        embedded = OpenAIEmbedder().embed_many(chunks)
    except Exception:
        log.exception("embedding_ingest_failed", chunk_count=len(chunks))
        raise HTTPException(status_code=500, detail="embedding generation failed") from None

    document = Document(
        source_path=request.source_path,
        document_type=request.document_type,
        doc_metadata={},
    )
    session.add(document)
    await session.flush()

    session.add_all(
        ChunkRow(
            document_id=document.id,
            chunk_type=CHUNK_TYPE_BUDGET_COMPONENT,
            content=chunk.text,
            embedding=chunk.embedding,
            chunk_metadata=chunk.metadata,
        )
        for chunk in embedded
    )
    await session.commit()

    embedding_dimension = len(embedded[0].embedding) if embedded else 0
    ingestion_time_ms = int((time.perf_counter() - t0) * 1000)
    return IngestDocumentResponse(
        document_id=document.id,
        chunks_created=len(embedded),
        embedding_dimension=embedding_dimension,
        ingestion_time_ms=ingestion_time_ms,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest, session: AsyncSession = Depends(get_db_session)
) -> SearchResponse:
    t0 = time.perf_counter()

    query_vector = OpenAIEmbedder().embed_one(request.query)

    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.chunk_metadata,
            distance,
        )
        .order_by(distance)
        .limit(request.k)
    )
    rows = (await session.execute(stmt)).all()

    search_time_ms = int((time.perf_counter() - t0) * 1000)
    return SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=search_time_ms,
        results=[
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=row.distance,
                metadata=row.chunk_metadata,
            )
            for row in rows
        ],
    )
