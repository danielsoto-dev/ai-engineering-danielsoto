"""Tests for graph thread configuration and API response adaptation."""

from __future__ import annotations

from app.graph.runner import checkpoint_database_url, invoke_estimation_graph
from app.graph.state import EstimationGraphContext


class _FakeGraph:
    config = None

    async def ainvoke(self, input, config, *, context):
        self.config = config
        return {
            **input,
            "estimate": {
                "components": [
                    {
                        "name": "Backend",
                        "estimated_hours": 115,
                        "reference_chunk_ids": [10],
                        "rationale": "Historical median plus contingency.",
                    }
                ],
                "total_hours": 115,
                "assumptions": [],
                "confidence": "medium",
            },
            "status": "validated",
            "errors": [],
            "trace_steps": [
                {
                    "step": 1,
                    "reasoning": "Extract requirements.",
                    "action": "extract_requirements",
                    "arguments": {"transcript_chars": 30},
                    "observation": "extracted 1 requirement",
                }
            ],
        }


class _FakeClient:
    pass


def test_checkpoint_url_removes_sqlalchemy_driver() -> None:
    assert (
        checkpoint_database_url("postgresql+asyncpg://user:pass@localhost:5433/database")
        == "postgresql://user:pass@localhost:5433/database"
    )


async def test_runner_passes_estimation_id_as_thread_id() -> None:
    graph = _FakeGraph()
    result = await invoke_estimation_graph(
        graph,
        transcript="Build a logistics backend API.",
        estimation_id="estimate-123",
        context=EstimationGraphContext(client=_FakeClient()),
    )

    assert graph.config == {"configurable": {"thread_id": "estimate-123"}}
    assert result.estimation_id == "estimate-123"
    assert result.status == "validated"
