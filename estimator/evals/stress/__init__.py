"""Session 6 stress test — measure where the CAG breaks.

Instruments the conversational pipeline and runs it through three load
scenarios (long multi-turn, large attachments, contradiction) to produce a
quantitative baseline: latency vs tokens, cumulative cost vs turn, and fact
recall vs history length. The deliverables are ``results.csv`` + ``REPORT.md``.

Modules:
- ``scenarios.py``: the three multi-turn profiles + their fact-trackers.
- ``fixtures/build_pdfs.py``: regenerates the synthetic attachment corpus.
- ``metrics.py``: LatencyBudget / CostBudget / MemoryDrift metrics.
- ``run.py``: the CLI runner that drives the sessions API and writes the CSV.
"""

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import SCENARIOS, Scenario, Turn

__all__ = [
    "SCENARIOS",
    "Scenario",
    "Turn",
    "LatencyBudgetMetric",
    "CostBudgetMetric",
    "MemoryDriftMetric",
]
