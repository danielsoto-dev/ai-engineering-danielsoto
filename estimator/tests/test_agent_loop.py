"""Tests for the manual Responses API agent loop."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.agent.loop import run_estimation_agent
from app.agent.schemas import HistoricalBudgetItem, SearchBudgetsArgs


def _reasoning(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _tool_call(name: str, call_id: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=json.dumps(arguments),
    )


class FakeResponses:
    def __init__(self, outputs: list[list[SimpleNamespace]], final_output: str) -> None:
        self.outputs = outputs
        self.final_output = final_output
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        output = self.outputs[min(index, len(self.outputs) - 1)]
        output_text = (
            self.final_output if not any(i.type == "function_call" for i in output) else ""
        )
        return SimpleNamespace(id=f"response_{index + 1}", output=output, output_text=output_text)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


async def fake_retrieval(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
    return [
        HistoricalBudgetItem(
            id=10,
            content_preview=args.query,
            sector="logistics",
            budget_id="BUD-10",
            estimated_hours=100.0,
        )
    ]


def _final_output() -> str:
    return json.dumps(
        {
            "components": [
                {
                    "name": "Backend",
                    "estimated_hours": 115.0,
                    "reference_chunk_ids": [10],
                    "rationale": "Median historical effort plus contingency.",
                }
            ],
            "total_hours": 115.0,
            "assumptions": [],
            "confidence": "medium",
        }
    )


def _happy_path() -> list[list[SimpleNamespace]]:
    return [
        [
            _reasoning("The project has two distinct components."),
            _tool_call("search_budgets", "search_1", {"query": "backend", "filters": None}),
            _tool_call("search_budgets", "search_2", {"query": "mobile app", "filters": None}),
        ],
        [
            _reasoning("Both components now have references."),
            _tool_call(
                "calculate_estimate",
                "calculate_1",
                {"components": [{"name": "Backend", "reference_amounts": [100.0]}]},
            ),
        ],
        [SimpleNamespace(type="message")],
    ]


async def test_agent_runs_multiple_tools_and_returns_structured_estimate() -> None:
    responses = FakeResponses(_happy_path(), _final_output())
    result = await run_estimation_agent(
        "A backend and mobile app",
        client=FakeClient(responses),
        model="gpt-5-mini",
        retrieval_backend=fake_retrieval,
    )

    actions = [step.action for step in result.trace.steps]
    assert actions.count("search_budgets") == 2
    assert "calculate_estimate" in actions
    assert all(step.reasoning and step.observation for step in result.trace.steps)
    assert result.stop_reason == "completed"
    assert result.estimate is not None
    assert result.estimate.total_hours == 115.0


async def test_agent_returns_outputs_with_matching_call_ids() -> None:
    responses = FakeResponses(_happy_path(), _final_output())
    await run_estimation_agent(
        "A backend and mobile app",
        client=FakeClient(responses),
        retrieval_backend=fake_retrieval,
    )

    returned_outputs = responses.calls[1]["input"]
    assert {output["call_id"] for output in returned_outputs} == {"search_1", "search_2"}
    assert all(output["type"] == "function_call_output" for output in returned_outputs)


async def test_agent_stops_at_max_iterations() -> None:
    repeated_call = [
        _tool_call("search_budgets", "search_forever", {"query": "backend", "filters": None})
    ]
    responses = FakeResponses([repeated_call], _final_output())
    result = await run_estimation_agent(
        "A backend",
        client=FakeClient(responses),
        max_iterations=3,
        retrieval_backend=fake_retrieval,
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert result.estimate is None


async def test_non_object_tool_arguments_become_an_observation() -> None:
    outputs = [
        [
            SimpleNamespace(
                type="function_call",
                name="search_budgets",
                call_id="bad_arguments",
                arguments="[]",
            )
        ],
        [SimpleNamespace(type="message")],
    ]
    responses = FakeResponses(outputs, _final_output())
    result = await run_estimation_agent(
        "A backend",
        client=FakeClient(responses),
        retrieval_backend=fake_retrieval,
    )

    assert result.stop_reason == "completed"
    assert "JSON object" in result.trace.steps[0].observation
    assert result.trace.steps[0].arguments == {}
