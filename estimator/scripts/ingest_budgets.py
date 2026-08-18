"""Ingest a budgets JSON file through POST /embeddings/ingest.

Usage:
    uv run python scripts/ingest_budgets.py data/budgets_extended.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"


def main(path: str) -> int:
    budgets = json.loads(Path(path).read_text(encoding="utf-8"))
    ingested = skipped = 0

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        for budget in budgets:
            source_path = f"{path.split('/')[-1]}#{budget['budget_id']}"
            response = client.post(
                "/embeddings/ingest",
                json={
                    "source_path": source_path,
                    "document_type": "historical_budget",
                    "content": budget,
                },
            )
            if response.status_code == 409:
                skipped += 1
                print(f"skip   {source_path} (already ingested)")
                continue
            if response.status_code != 200:
                print(f"FAIL   {source_path}: {response.status_code} {response.text[:200]}")
                return 1
            body = response.json()
            ingested += 1
            print(f"ok     {source_path}: {body['chunks_created']} chunks")

    print(f"\ningested={ingested} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/budgets_extended.json"))
