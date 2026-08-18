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
from app.retrieval.hybrid import lexical_search, reciprocal_rank_fusion, vector_search
from app.retrieval.reranker import rerank

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

    # Reranking only pays off if it gets more candidates than the caller wants
    # back: recall wide, then let the cross-encoder pick.
    retrieval_limit = max(request.candidate_k, request.k) if request.rerank else request.k

    query_vector = OpenAIEmbedder().embed_one(request.query)
    ranked = await vector_search(session, query_vector, retrieval_limit)

    if request.mode == "hybrid":
        lexical = await lexical_search(session, request.query, retrieval_limit)
        ranked = reciprocal_rank_fusion([ranked, lexical], k=request.rrf_k)

    candidates_considered = len(ranked)

    if request.rerank:
        ranked = rerank(request.query, ranked, top_k=request.k)
    else:
        ranked = ranked[: request.k]

    search_time_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "search_completed",
        mode=request.mode,
        reranked=request.rerank,
        candidates=candidates_considered,
        returned=len(ranked),
        search_time_ms=search_time_ms,
    )
    return SearchResponse(
        query=request.query,
        k=request.k,
        mode=request.mode,
        reranked=request.rerank,
        candidates_considered=candidates_considered,
        search_time_ms=search_time_ms,
        results=[
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_type=chunk.chunk_type,
                content=chunk.content,
                distance=chunk.vector_distance,
                metadata=chunk.metadata,
                lexical_rank_score=chunk.lexical_rank_score,
                fusion_score=chunk.fusion_score,
                rerank_score=chunk.rerank_score,
                sources=chunk.sources,
            )
            for chunk in ranked
        ],
    )
