"""Retrieval-to-generation seam: fetch chunks and render them for the prompt.

The generator may only cite chunks it was actually shown, so this module is the
single place that decides that set. ``assemble_context`` returns both the prompt
block and the ids it contains, and ``verify_citations`` is checked against those
same ids — never against a fresh query, which could drift and silently bless a
citation the model never saw.

Search defaults to hybrid without reranking: on the Session 10 golden set that
scored P@5 0.84 against 0.72 for hybrid + cross-encoder, so reranking is off
unless a caller asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.retrieval.hybrid import (
    RetrievedChunk,
    lexical_search,
    reciprocal_rank_fusion,
    vector_search,
)

log = structlog.get_logger()

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 50


@dataclass
class AssembledContext:
    """A rendered context block plus the exact chunk ids it exposes."""

    text: str
    chunk_ids: set[str]
    chunks: list[RetrievedChunk]

    @property
    def contexts(self) -> list[str]:
        """Chunk contents, the shape RAGAS expects for ``contexts``."""
        return [chunk.content for chunk in self.chunks]

    @property
    def sorted_chunk_ids(self) -> list[str]:
        """Ids in numeric order. Plain ``sorted`` compares them as strings and
        renders ``['11', '4']``, which reads as a mistake in logs and reports."""
        return sorted(self.chunk_ids, key=int)


def format_chunk(chunk: RetrievedChunk) -> str:
    """Label a chunk with the id the model must cite."""
    return f"[chunk_id: {chunk.chunk_id} | document_id: {chunk.document_id}]\n{chunk.content}"


def render_context(chunks: list[RetrievedChunk]) -> AssembledContext:
    """Render already-retrieved chunks. Split out so tests can build a context
    without a database."""
    return AssembledContext(
        text="\n\n---\n\n".join(format_chunk(chunk) for chunk in chunks),
        chunk_ids={str(chunk.chunk_id) for chunk in chunks},
        chunks=chunks,
    )


async def assemble_context(
    session,
    query: str,
    query_vector: list[float],
    *,
    top_k: int = DEFAULT_TOP_K,
    use_rerank: bool = False,
) -> AssembledContext:
    """Retrieve the top chunks for ``query`` and render them for the prompt."""
    limit = DEFAULT_CANDIDATE_K if use_rerank else top_k

    ranked = await vector_search(session, query_vector, limit)
    lexical = await lexical_search(session, query, limit)
    ranked = reciprocal_rank_fusion([ranked, lexical])

    if use_rerank:
        from app.retrieval.reranker import rerank

        ranked = rerank(query, ranked, top_k=top_k)
    else:
        ranked = ranked[:top_k]

    context = render_context(ranked)
    log.info(
        "context_assembled",
        chunk_ids=context.sorted_chunk_ids,
        rerank=use_rerank,
        chunks=len(ranked),
    )
    return context
