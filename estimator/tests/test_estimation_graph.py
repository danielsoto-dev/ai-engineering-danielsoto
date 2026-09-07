"""Tests for the sequential Session 13 LangGraph workflow."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.agent.schemas import HistoricalBudgetItem, SearchBudgetsArgs
from app.graph import EstimationGraphContext, build_estimation_graph


class _FakeResponses:
    async def create(self, **kwargs):
        format_name = kwargs["text"]["format"]["name"]
        if format_name == "requirement_extraction":
            body = {
                "requirements": [
                    "Create a business API",
                    "Synchronize billing with SAP",
                ]
            }
        else:
            body = {
                "components": [
                    {
                        "name": "Business API",
                        "category": "backend",
                        "search_query": "logistics business API",
                    },
                    {
                        "name": "SAP integration",
                        "category": "integration",
                        "search_query": "SAP billing integration",
                    },
                ]
            }
        return SimpleNamespace(output_text=json.dumps(body))


class _FakeClient:
    responses = _FakeResponses()


async def _fake_retrieval(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
    item_id = 10 if "logistics" in args.query else 20
    return [
        HistoricalBudgetItem(
            id=item_id,
            content_preview=args.query,
            sector="logistics",
            budget_id=f"BUD-{item_id}",
            estimated_hours=100.0,
        )
    ]


def _initial_state() -> dict:
    return {
        "estimation_id": "estimate-test-1",
        "transcript": "Build a logistics API and synchronize billing with SAP.",
        "requirements": [],
        "components": [],
        "budget_matches": [],
        "estimate": None,
        "status": None,
        "errors": [],
        "trace_steps": [],
    }


async def test_graph_runs_all_five_nodes_sequentially() -> None:
    graph = build_estimation_graph()
    result = await graph.ainvoke(
        _initial_state(),
        context=EstimationGraphContext(
            client=_FakeClient(),
            model="gpt-5-mini",
            reasoning_effort="minimal",
            retrieval_backend=_fake_retrieval,
        ),
    )

    assert result["status"] == "validated"
    assert result["estimate"]["total_hours"] == 230.0
    assert [step["action"] for step in result["trace_steps"]] == [
        "extract_requirements",
        "classify_components",
        "search_budgets",
        "generate_estimate",
        "validate_and_consolidate",
    ]
    assert len(result["budget_matches"]) == 2


async def test_graph_marks_missing_references_for_review() -> None:
    async def no_results(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
        return []

    result = await build_estimation_graph().ainvoke(
        _initial_state(),
        context=EstimationGraphContext(
            client=_FakeClient(),
            retrieval_backend=no_results,
        ),
    )

    assert result["status"] == "needs_review"
    assert len(result["errors"]) == 4
