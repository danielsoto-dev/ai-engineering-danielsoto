"""Stress-test metrics (Block 4).

These live in ``evals/stress`` rather than ``evals/metrics.py`` on purpose:
the Session-5 metrics score an ``EstimationResult`` (the model's answer),
while these score the *operational* shape of a turn — latency, cost, and
whether a fact survived in the session snapshot. Different input type, same
``MetricResult`` contract, so they compose with the existing harness without
bending ``run_all_metrics`` to two signatures.

Determinism over sophistication: budgets are plain comparisons and memory
recall is a case-insensitive substring match over the snapshot. No embeddings,
no LLM-as-judge — a red metric must be reproducible from the CSV alone.
"""

from __future__ import annotations

from typing import Any

# Reuse the exact dataclass the Session-5 harness defines, so a stress
# MetricResult is indistinguishable from a golden-eval one downstream.
from evals.metrics import MetricResult


class LatencyBudgetMetric:
    """1.0 if ``latency_ms <= budget_ms``; 0.0 otherwise.

    Turns an SLA into a test: the latency budget is a design contract, not a
    number you eyeball after the fact.
    """

    def __init__(self, budget_ms: int) -> None:
        self.budget_ms = budget_ms
        self.name = "latency_budget"

    def evaluate(self, observation: dict[str, Any]) -> MetricResult:
        latency = float(observation.get("latency_ms", 0))
        passed = latency <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"latency_ms={latency:.0f} budget_ms={self.budget_ms}",
        )


class CostBudgetMetric:
    """1.0 if ``cost_usd <= budget_usd``; 0.0 otherwise."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.name = "cost_budget"

    def evaluate(self, observation: dict[str, Any]) -> MetricResult:
        cost = float(observation.get("cost_usd", 0.0))
        passed = cost <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"cost_usd={cost:.6f} budget_usd={self.budget_usd}",
        )


class MemoryDriftMetric:
    """1.0 if a fact introduced at turn k is still present at a later turn.

    Looks for ``fact`` (case-insensitive substring) across the chosen
    locations of a session snapshot:

    - ``summary``  -> snapshot["summary_text"]   (the cumulative summary)
    - ``anchors``  -> snapshot["anchors_text"]    (joined anchor contents)
    - ``metadata`` -> snapshot["metadata_text"]   (joined ProjectMetadata)

    The runner builds these flattened text fields from ``GET /sessions/{id}``.
    A miss is the headline signal of the exercise: the point where the CAG
    silently forgets.
    """

    _VALID = ("summary", "anchors", "metadata")

    def __init__(self, fact: str, where: list[str] | None = None) -> None:
        self.fact = fact
        self.where = where or list(self._VALID)
        bad = [w for w in self.where if w not in self._VALID]
        if bad:
            raise ValueError(f"unknown location(s) {bad}; valid: {self._VALID}")
        self.name = "memory_drift"

    def evaluate(self, snapshot: dict[str, Any]) -> MetricResult:
        needle = self.fact.lower()
        found_in = [
            loc
            for loc in self.where
            if needle in str(snapshot.get(f"{loc}_text", "")).lower()
        ]
        passed = bool(found_in)
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"fact={self.fact!r} found_in={found_in}"
                if passed
                else f"fact={self.fact!r} NOT FOUND in {self.where}"
            ),
        )
