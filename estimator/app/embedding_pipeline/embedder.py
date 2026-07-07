"""OpenAI embedder for the embedding pipeline (Session 7 pre-exercise, Step 4)."""

from __future__ import annotations

import time

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
# $/million input tokens, text-embedding-3-small (OpenAI pricing, subject to change).
PRICE_PER_MILLION_TOKENS_USD = 0.02
_RETRY_DELAYS_S = (1, 2, 4)


class OpenAIEmbedder:
    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client or OpenAI()

    def embed_one(self, text: str) -> list[float]:
        return self._create_with_retry([text])[0].embedding

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            t0 = time.perf_counter()
            data = self._create_with_retry([c.text for c in batch])
            latency_ms = int((time.perf_counter() - t0) * 1000)
            total_tokens = sum(c.token_count for c in batch)
            log.info(
                "embedding_batch_completed",
                batch_size=len(batch),
                total_tokens=total_tokens,
                latency_ms=latency_ms,
            )
            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=item.embedding)
                for chunk, item in zip(batch, data)
            )
        return embedded

    def _create_with_retry(self, inputs: list[str]):
        for attempt, delay in enumerate((*_RETRY_DELAYS_S, None)):
            try:
                response = self._client.embeddings.create(model=EMBEDDING_MODEL, input=inputs)
                return response.data
            except RateLimitError:
                if delay is None:
                    raise
                log.warning("embedding_rate_limited", attempt=attempt, retry_in_s=delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover


def estimate_cost_usd(total_tokens: int) -> float:
    return (total_tokens / 1_000_000) * PRICE_PER_MILLION_TOKENS_USD
