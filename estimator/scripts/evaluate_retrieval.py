"""Measure P@5 and latency for the four retrieval configurations.

    uv run python scripts/evaluate_retrieval.py

Configurations, per the Session 10 exercise:

    A  vector  no rerank
    B  hybrid  no rerank
    C  vector  rerank
    D  hybrid  rerank

Runs against the database directly rather than over HTTP, so the reported
latency is retrieval plus reranking without web-server overhead.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing anything that reads settings at import time. The
# script runs on the host, so the container hostname in DATABASE_URL does not
# resolve — point it at the port compose publishes unless the caller overrode it.
load_dotenv()
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://estimator:estimator@localhost:5433/estimator"
)
if "@postgres:" in os.environ.get("DATABASE_URL", ""):
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace(
        "@postgres:5432", "@localhost:5433"
    )

from app.db.session import get_sessionmaker  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402
from app.retrieval.hybrid import (  # noqa: E402
    RetrievedChunk,
    lexical_search,
    reciprocal_rank_fusion,
    vector_search,
)
from app.retrieval.reranker import rerank  # noqa: E402

GOLDEN_SET_PATH = Path("evals/retrieval_golden_set.json")
REPORT_PATH = Path("evals/retrieval_results.md")

TOP_K = 5
CANDIDATE_K = 50

CONFIGURATIONS = [
    ("A", "vector", False),
    ("B", "hybrid", False),
    ("C", "vector", True),
    ("D", "hybrid", True),
]


async def run_query(
    session, query: str, mode: str, use_rerank: bool
) -> tuple[list[int], float, int]:
    """Return (top-k chunk ids, elapsed ms, candidates considered)."""
    t0 = time.perf_counter()

    limit = CANDIDATE_K if use_rerank else TOP_K
    ranked = await vector_search(session, OpenAIEmbedder().embed_one(query), limit)

    if mode == "hybrid":
        lexical = await lexical_search(session, query, limit)
        ranked = reciprocal_rank_fusion([ranked, lexical])

    candidates = len(ranked)
    ranked = rerank(query, ranked, top_k=TOP_K) if use_rerank else ranked[:TOP_K]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [c.chunk_id for c in ranked], elapsed_ms, candidates


def precision_at_k(retrieved: list[int], relevant: set[int]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for chunk_id in retrieved if chunk_id in relevant) / len(retrieved)


async def main() -> int:
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]

    # Warm the cross-encoder so the first measured query does not pay model load.
    # rerank() short-circuits on an empty candidate list, so warm it with a real one.
    rerank("warmup", [RetrievedChunk(0, 0, "warmup", "warmup", {})], top_k=1)

    results: dict[str, dict] = {}
    per_query: list[dict] = []

    async with get_sessionmaker()() as session:
        for label, mode, use_rerank in CONFIGURATIONS:
            precisions: list[float] = []
            latencies: list[float] = []

            for case in queries:
                relevant = set(case["relevant_chunk_ids"])
                retrieved, elapsed_ms, candidates = await run_query(
                    session, case["query"], mode, use_rerank
                )
                precision = precision_at_k(retrieved, relevant)
                precisions.append(precision)
                latencies.append(elapsed_ms)
                per_query.append(
                    {
                        "config": label,
                        "query_id": case["id"],
                        "precision_at_5": precision,
                        "latency_ms": round(elapsed_ms, 1),
                        "candidates": candidates,
                        "retrieved": retrieved,
                        "hits": [c for c in retrieved if c in relevant],
                    }
                )
                print(
                    f"{label} {case['id']}  P@5={precision:.2f}  "
                    f"{elapsed_ms:7.1f}ms  candidates={candidates}"
                )

            results[label] = {
                "mode": mode,
                "rerank": use_rerank,
                "mean_precision_at_5": statistics.mean(precisions),
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
            }
            print()

    _write_report(results, per_query, queries)
    print(f"report written to {REPORT_PATH}")
    return 0


def _write_report(results: dict, per_query: list[dict], queries: list[dict]) -> None:
    lines = [
        "# Retrieval evaluation — Session 10",
        "",
        f"Golden set: {len(queries)} queries, criterion `functional_domain`. "
        f"top-k = {TOP_K}, recall depth for reranking = {CANDIDATE_K}.",
        "",
        "## Comparative table",
        "",
        "| Config | Búsqueda | Reranking | P@5 medio | Latencia media (ms) | Latencia mediana (ms) |",
        "| ------ | -------- | --------- | --------- | ------------------- | --------------------- |",
    ]
    search_label = {"vector": "Vectorial", "hybrid": "Híbrida"}
    for label, data in results.items():
        lines.append(
            f"| {label} | {search_label[data['mode']]} | {'Sí' if data['rerank'] else 'No'} "
            f"| {data['mean_precision_at_5']:.2f} "
            f"| {data['mean_latency_ms']:.0f} "
            f"| {data['median_latency_ms']:.0f} |"
        )

    lines += [
        "",
        "## Per-query detail",
        "",
        "| Config | Query | P@5 | Latencia (ms) | Candidatos |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in per_query:
        lines.append(
            f"| {row['config']} | {row['query_id']} | {row['precision_at_5']:.2f} "
            f"| {row['latency_ms']:.0f} | {row['candidates']} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
