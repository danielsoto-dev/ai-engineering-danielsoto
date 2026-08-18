"""Cross-encoder reranker for the recall-then-rerank pattern.

A bi-encoder embeds the query and the document separately, so it never sees
them together. A cross-encoder scores the pair in one forward pass and can tell
that "plataforma de e-commerce" and "app de pagos" share vocabulary but not
intent. That accuracy costs a model call per candidate, which is why it runs
over a shallow top-N rather than the whole corpus.
"""

from __future__ import annotations

import structlog

from app.retrieval.hybrid import RetrievedChunk

log = structlog.get_logger()

# Multilingual cross-encoder — the corpus and the queries are in Spanish, so an
# English-only reranker would score on the wrong language.
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model_cache: dict[str, object] = {}


def _load_model(model_name: str):
    """Load and cache the cross-encoder.

    Import is deferred so the app starts without paying the torch import cost
    when reranking is switched off.
    """
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached

    from sentence_transformers import CrossEncoder

    log.info("reranker_loading", model=model_name)
    model = CrossEncoder(model_name)
    _model_cache[model_name] = model
    log.info("reranker_loaded", model=model_name)
    return model


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[RetrievedChunk]:
    """Score every (query, chunk) pair and return the best ``top_k``."""
    if not candidates:
        return []

    model = _load_model(model_name)
    scores = model.predict([(query, chunk.content) for chunk in candidates])

    for chunk, score in zip(candidates, scores, strict=True):
        chunk.rerank_score = float(score)

    return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]
