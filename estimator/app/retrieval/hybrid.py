"""Vector search, lexical search, and Reciprocal Rank Fusion over chunks.

The two branches answer different questions. Cosine distance over embeddings
finds chunks that *mean* something similar; ``ts_rank_cd`` over the Spanish
tsvector finds chunks that literally say the query's words. RRF merges the two
rankings without needing their scores to live on a comparable scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Float, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk as ChunkRow

# Smoothing constant from the original RRF paper. Large enough that the top few
# ranks score similarly, which keeps one branch from dominating the fusion.
DEFAULT_RRF_K = 60

TEXT_SEARCH_CONFIG = "spanish"


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    metadata: dict
    vector_distance: float | None = None
    lexical_rank_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    sources: list[str] = field(default_factory=list)


async def vector_search(
    session: AsyncSession, query_vector: list[float], limit: int
) -> list[RetrievedChunk]:
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
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            metadata=row.chunk_metadata,
            vector_distance=float(row.distance),
            sources=["vector"],
        )
        for row in rows
    ]


async def lexical_search(session: AsyncSession, query: str, limit: int) -> list[RetrievedChunk]:
    """Full-text search over the generated ``content_tsv`` column.

    Queries are natural-language project descriptions. Postgres' query parsers
    join terms with AND, so a whole sentence matches nothing — no chunk holds
    every word. OR-joined lexemes keep recall, and ``ts_rank_cd`` ranks by how
    many distinct terms a chunk covers.
    """
    tsquery = func.to_tsquery(
        TEXT_SEARCH_CONFIG,
        func.replace(func.plainto_tsquery(TEXT_SEARCH_CONFIG, query).cast(Text), " & ", " | "),
    )
    # ts_rank_cd weighs term density, so a chunk repeating a query term does not
    # outrank one that covers more distinct terms.
    rank = func.ts_rank_cd(ChunkRow.content_tsv, tsquery).cast(Float).label("rank")
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.chunk_metadata,
            rank,
        )
        .where(ChunkRow.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            metadata=row.chunk_metadata,
            lexical_rank_score=float(row.rank),
            sources=["lexical"],
        )
        for row in rows
    ]


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Merge ranked lists by summing 1 / (k + rank) per chunk.

    Only the position of a chunk in each list matters, never its raw score —
    that is what lets a cosine distance and a ts_rank_cd value be combined at
    all. A chunk found by both branches accumulates two contributions and
    therefore outranks one found by a single branch at the same position.
    """
    merged: dict[int, RetrievedChunk] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            contribution = 1.0 / (k + position)
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_type=chunk.chunk_type,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    vector_distance=chunk.vector_distance,
                    lexical_rank_score=chunk.lexical_rank_score,
                    fusion_score=contribution,
                    sources=list(chunk.sources),
                )
                continue
            existing.fusion_score = (existing.fusion_score or 0.0) + contribution
            # Keep whichever branch-specific score each list carried.
            if chunk.vector_distance is not None:
                existing.vector_distance = chunk.vector_distance
            if chunk.lexical_rank_score is not None:
                existing.lexical_rank_score = chunk.lexical_rank_score
            for source in chunk.sources:
                if source not in existing.sources:
                    existing.sources.append(source)

    return sorted(merged.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)
