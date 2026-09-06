"""Unit tests for the Session 12 agent tools."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.schemas import HistoricalBudgetItem, SearchBudgetsArgs
from app.agent.tools import (
    CALCULATE_ESTIMATE_TOOL,
    CONTINGENCY_FACTOR,
    SEARCH_BUDGETS_TOOL,
    calculate_estimate,
    dispatch_tool,
    search_budgets,
)


def test_tool_schemas_are_flat_and_strict() -> None:
    for schema in (SEARCH_BUDGETS_TOOL, CALCULATE_ESTIMATE_TOOL):
        assert schema["type"] == "function"
        assert schema["strict"] is True
        assert "name" in schema
        assert "function" not in schema
        assert schema["parameters"]["additionalProperties"] is False


def test_calculate_estimate_uses_median_and_contingency() -> None:
    result = calculate_estimate(
        {"components": [{"name": "Backend", "reference_amounts": [100.0, 200.0, 900.0]}]}
    )

    assert result["components"][0]["estimated_hours"] == pytest.approx(
        200.0 * (1 + CONTINGENCY_FACTOR)
    )
    assert result["components"][0]["unbudgeted"] is False
    assert result["total_hours"] == 230.0


def test_calculate_estimate_flags_missing_references() -> None:
    result = calculate_estimate(
        {"components": [{"name": "Unknown integration", "reference_amounts": []}]}
    )

    assert result["components"][0]["estimated_hours"] == 0.0
    assert result["components"][0]["unbudgeted"] is True


def test_calculate_estimate_rejects_invalid_arguments() -> None:
    with pytest.raises(ValidationError):
        calculate_estimate({"components": [{"name": "Backend"}]})


async def test_search_budgets_uses_injected_backend() -> None:
    async def backend(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
        assert args.query == "SAP integration"
        return [
            HistoricalBudgetItem(
                id=3,
                content_preview="SAP integration",
                sector="industrial",
                budget_id="BUD-3",
                estimated_hours=400.0,
            )
        ]

    result = await search_budgets({"query": "SAP integration", "filters": None}, backend=backend)

    assert result["count"] == 1
    assert result["items"][0]["estimated_hours"] == 400.0
    assert "400.0" in result["summary"]


async def test_dispatch_rejects_unknown_tool() -> None:
    async def backend(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
        return []

    with pytest.raises(ValueError, match="Unknown tool"):
        await dispatch_tool("invented_tool", {}, retrieval_backend=backend)
