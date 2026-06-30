"""Unit tests for the Session 6 stress metrics (Block 4).

One passing case, one failing case, and one boundary case per metric — the
minimum the exercise asks for. Pure functions over plain dicts, no LLM.
"""

from __future__ import annotations

import pytest

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)


# --- LatencyBudgetMetric -----------------------------------------------------


def test_latency_under_budget_passes():
    m = LatencyBudgetMetric(budget_ms=4000)
    r = m.evaluate({"latency_ms": 1200})
    assert r.passed and r.score == 1.0


def test_latency_over_budget_fails():
    m = LatencyBudgetMetric(budget_ms=4000)
    r = m.evaluate({"latency_ms": 9000})
    assert not r.passed and r.score == 0.0


def test_latency_exactly_at_budget_passes():
    # Boundary: <= is inclusive, so equal is a pass.
    m = LatencyBudgetMetric(budget_ms=4000)
    assert m.evaluate({"latency_ms": 4000}).passed


# --- CostBudgetMetric --------------------------------------------------------


def test_cost_under_budget_passes():
    m = CostBudgetMetric(budget_usd=0.01)
    assert m.evaluate({"cost_usd": 0.002}).passed


def test_cost_over_budget_fails():
    m = CostBudgetMetric(budget_usd=0.01)
    assert not m.evaluate({"cost_usd": 0.05}).passed


def test_cost_missing_field_defaults_zero_and_passes():
    # Boundary: a turn with no usage reported reads as 0.0 cost -> passes.
    m = CostBudgetMetric(budget_usd=0.01)
    assert m.evaluate({}).passed


# --- MemoryDriftMetric -------------------------------------------------------


def test_memory_fact_present_in_summary_passes():
    m = MemoryDriftMetric(fact="Nimbus")
    snap = {"summary_text": "The project Nimbus is a CRM.", "anchors_text": "", "metadata_text": ""}
    r = m.evaluate(snap)
    assert r.passed and "summary" in r.details


def test_memory_fact_absent_everywhere_fails():
    m = MemoryDriftMetric(fact="Nimbus")
    snap = {"summary_text": "Some other project.", "anchors_text": "", "metadata_text": ""}
    assert not m.evaluate(snap).passed


def test_memory_match_is_case_insensitive():
    # Boundary: stored lowercase, fact mixed-case -> still a hit.
    m = MemoryDriftMetric(fact="Flutter", where=["metadata"])
    assert m.evaluate({"metadata_text": "stack includes flutter and dart"}).passed


def test_memory_respects_where_scope():
    # Fact only in summary, but we only look in anchors -> miss.
    m = MemoryDriftMetric(fact="Atlas", where=["anchors"])
    snap = {"summary_text": "Atlas is the tool.", "anchors_text": ""}
    assert not m.evaluate(snap).passed


def test_memory_rejects_unknown_location():
    with pytest.raises(ValueError):
        MemoryDriftMetric(fact="x", where=["nonsense"])
