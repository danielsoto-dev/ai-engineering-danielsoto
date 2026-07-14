#!/usr/bin/env python3
"""Ingest the sample budget corpus and run 5 representative queries against
/search.

uv run python scripts/query_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402

BASE_URL = get_settings().ESTIMATOR_API_BASE_URL
CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"

QUERIES = [
    "REST API development with JWT authentication for financial sector",
    "secure backend service with token-based access control for banking applications",
    "mobile application for restaurant reservations",
    "integration with external system",
    "migration from monolith to microservices architecture using Kubernetes",
]


def ingest_corpus(client: httpx.Client) -> None:
    budgets = json.loads(CORPUS_PATH.read_text())
    for budget in budgets:
        source_path = f"{CORPUS_PATH.name}#{budget['budget_id']}"
        response = client.post(
            "/embeddings/ingest",
            json={
                "source_path": source_path,
                "document_type": "historical_budget",
                "content": budget,
            },
        )
        if response.status_code == 409:
            print(f"[skip] {source_path} already ingested")
        else:
            response.raise_for_status()
            stats = response.json()
            print(f"[ingested] {source_path} -> {stats['chunks_created']} chunks")


def run_queries(client: httpx.Client) -> None:
    for query in QUERIES:
        response = client.post("/search", json={"query": query, "k": 5})
        response.raise_for_status()
        results = response.json()["results"]

        print(f"\nQuery: {query}")
        print("-" * 100)
        for r in results:
            snippet = r["content"][:120].replace("\n", " ")
            print(
                f"  chunk_id={r['chunk_id']:<5} distance={r['distance']:.4f} "
                f"type={r['chunk_type']:<18} {snippet}"
            )


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        print("=== Ingesting sample corpus ===")
        ingest_corpus(client)
        print("\n=== Running example queries ===")
        run_queries(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
