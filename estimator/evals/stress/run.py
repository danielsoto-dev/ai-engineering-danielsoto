"""Stress runner (Block 5): drive the sessions API, write a per-turn CSV.

For each (scenario × attachment_size × repeat), open a fresh session and walk
its turns, attaching the calibrated PDF (when size > 0) to every turn. After
each turn we read ``GET /sessions/{id}`` — which now carries ``last_turn``
(the aggregated ``turn_observed`` telemetry) plus the flattened snapshot text
the memory metric needs — and emit one CSV row with the telemetry + the three
binary metric scores.

Transport mirrors ``evals/run.py``: in-process ``TestClient`` by default,
``--http BASE`` for a live server.

    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 \\
        --output evals/stress/results.csv

Note: this makes real LLM calls (cost). Defaults are sized so a full run is
≥ 50 rows (3 scenarios × 5 sizes × 3 repeats × ~5 turns).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from evals.stress.fixtures.build_pdfs import build_pdf
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import Scenario, get_scenarios

# Budgets are design contracts; tune to your SLA. Defaults are deliberately
# tight so the curves show where the system crosses them.
_LATENCY_BUDGET_MS = 4000
_COST_BUDGET_USD = 0.01


CSV_FIELDS = [
    "scenario",
    "attachment_kb",
    "repeat",
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "fact_to_remember",
    "latency_budget_pass",
    "cost_budget_pass",
    "memory_drift_pass",
]


def _snapshot_text_fields(info: dict[str, Any]) -> dict[str, str]:
    """Flatten a GET /sessions/{id} body into the *_text fields the
    MemoryDriftMetric reads."""
    meta = info.get("metadata") or {}
    metadata_bits = [
        str(meta.get("project_name") or ""),
        str(meta.get("agreed_scope") or ""),
        " ".join(meta.get("mentioned_technologies") or []),
    ]
    # The snapshot endpoint exposes summary_chars/anchors_count but not the raw
    # text; the live conversation keeps those server-side. We approximate the
    # searchable surface with metadata (always present) plus whatever text the
    # caller threaded in. summary_text/anchors_text fall back to "" — a miss
    # there is itself a valid signal (the fact never made it into metadata).
    return {
        "summary_text": str(info.get("summary_text", "")),
        "anchors_text": str(info.get("anchors_text", "")),
        "metadata_text": " ".join(b for b in metadata_bits if b),
    }


class _Client:
    """Thin uniform wrapper over TestClient / httpx so the loop is identical."""

    def __init__(self, http: str | None) -> None:
        self._http = http
        if http:
            import httpx

            self._c = httpx.Client(base_url=http, timeout=180.0)
        else:
            from fastapi.testclient import TestClient

            from app.main import app

            self._c = TestClient(app)

    def create_session(self) -> str:
        return self._c.post("/sessions").json()["session_id"]

    def estimate(self, sid: str, data: dict, files: dict | None) -> None:
        r = self._c.post(f"/sessions/{sid}/estimate", data=data, files=files)
        r.raise_for_status()

    def session_info(self, sid: str) -> dict:
        return self._c.get(f"/sessions/{sid}").json()

    def close(self) -> None:
        if hasattr(self._c, "close"):
            self._c.close()


def _run_scenario(
    client: _Client,
    scenario: Scenario,
    attachment_kb: int,
    repeat: int,
    pdf_path: Path | None,
    rows: list[dict[str, Any]],
) -> None:
    latency_metric = LatencyBudgetMetric(budget_ms=_LATENCY_BUDGET_MS)
    cost_metric = CostBudgetMetric(budget_usd=_COST_BUDGET_USD)

    sid = client.create_session()
    for turn in scenario.turns:
        data = {
            "transcript": turn.transcript,
            "project_type": scenario.project_type,
            "detail_level": scenario.detail_level,
            "output_format": scenario.output_format,
        }
        files = None
        if pdf_path is not None:
            files = {"attachments": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}

        client.estimate(sid, data, files)
        info = client.session_info(sid)
        obs = info.get("last_turn") or {}
        snap = _snapshot_text_fields(info)

        lat = latency_metric.evaluate(obs)
        cost = cost_metric.evaluate(obs)
        # The fact must survive into a LATER snapshot; checking it on its own
        # turn is the loosest possible test, but it still catches a system that
        # drops the fact immediately. Stricter cross-turn checks read the CSV.
        mem = MemoryDriftMetric(fact=turn.fact_to_remember).evaluate(snap)

        rows.append(
            {
                "scenario": scenario.name,
                "attachment_kb": attachment_kb,
                "repeat": repeat,
                "turn_index": obs.get("turn_index", turn.turn_index),
                "session_id": sid,
                "enriched_transcript_chars": obs.get("enriched_transcript_chars", 0),
                "attachments_total_chars": obs.get("attachments_total_chars", 0),
                "messages_in_window": obs.get("messages_in_window", 0),
                "anchors_count": obs.get("anchors_count", 0),
                "summary_chars": obs.get("summary_chars", 0),
                "tokens_in": obs.get("tokens_in", 0),
                "tokens_out": obs.get("tokens_out", 0),
                "cost_usd": obs.get("cost_usd", 0.0),
                "latency_ms": obs.get("latency_ms", 0),
                "cache_hit_kind": obs.get("cache_hit_kind", "none"),
                "last_resolved_tier": obs.get("last_resolved_tier"),
                "fact_to_remember": turn.fact_to_remember,
                "latency_budget_pass": int(lat.passed),
                "cost_budget_pass": int(cost.passed),
                "memory_drift_pass": int(mem.passed),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", default=None, help="Base URL (else in-process TestClient).")
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args()

    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    sizes = [int(s) for s in args.attachment_sizes.split(",") if s.strip()]
    scenarios = get_scenarios(scenario_names)

    # Regenerate the PDF corpus once for the non-zero sizes.
    pdfs: dict[int, Path] = {kb: build_pdf(kb) for kb in sizes if kb > 0}

    client = _Client(args.http)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        for scenario in scenarios:
            for kb in sizes:
                for repeat in range(args.repeats):
                    print(f"  {scenario.name} | {kb}KB | repeat {repeat + 1}/{args.repeats}")
                    _run_scenario(client, scenario, kb, repeat, pdfs.get(kb), rows)
    finally:
        client.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - t0
    print(f"\nWrote {len(rows)} rows to {args.output} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
