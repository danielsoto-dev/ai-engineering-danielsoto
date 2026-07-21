#!/usr/bin/env python3
"""Trace a transcript through the Session 08 embedding and search flow."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TRANSCRIPT = Path("examples/transcripts/02_ambiguous.txt")
DEFAULT_BASE_URL = "http://localhost:8000"
TOP_K = 5


def main() -> int:
    load_dotenv()
    transcript_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT
    if not transcript_path.is_file():
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    transcript = transcript_path.read_text(encoding="utf-8")
    response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=transcript)
    vector = response.data[0].embedding

    print("STEP 1 — EMBEDDING")
    print(f"model: {EMBEDDING_MODEL}")
    print(f"dimensions: {len(vector)}")
    print(f"norm: {math.sqrt(sum(value * value for value in vector)):.6f}")
    print(f"first_component: {vector[0]:.8f}")
    print(f"last_component: {vector[-1]:.8f}")

    base_url = os.getenv("ESTIMATOR_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    search_response = httpx.post(
        f"{base_url}/search",
        json={"query": transcript, "k": TOP_K},
        timeout=120,
    )
    search_response.raise_for_status()

    print("\nSTEP 2 — SEARCH")
    print(json.dumps(search_response.json(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
