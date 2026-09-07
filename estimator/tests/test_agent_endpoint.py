"""HTTP tests for the Session 12 agent endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.schemas import AgentEstimate, AgentTrace, GraphEstimationResult
from app.dependencies import get_agent_openai_client, get_estimation_graph_runner
from app.main import app


class _FakeRunner:
    async def run(self, *, transcript, estimation_id, context):
        return GraphEstimationResult(
            estimation_id=estimation_id,
            estimate=AgentEstimate.model_validate(
                {
                    "components": [
                        {
                            "name": "OAuth backend",
                            "estimated_hours": 184,
                            "reference_chunk_ids": [2, 3, 20],
                            "rationale": "Historical median plus contingency.",
                        }
                    ],
                    "total_hours": 184,
                    "assumptions": ["Existing Rails application."],
                    "confidence": "medium",
                }
            ),
            status="validated",
            errors=[],
            trace=AgentTrace(),
            iterations=5,
        )


class _FakeClient:
    pass


def test_agent_endpoint_returns_estimate_and_trace() -> None:
    app.dependency_overrides[get_agent_openai_client] = lambda: _FakeClient()
    app.dependency_overrides[get_estimation_graph_runner] = lambda: _FakeRunner()
    try:
        response = TestClient(app).post(
            "/api/v1/agent/estimate",
            json={"transcript": "Build an OAuth backend in Rails with PostgreSQL."},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["estimate"]["total_hours"] == 184
    assert body["trace"]["steps"] == []
    assert body["status"] == "validated"
    assert body["estimation_id"]
    assert body["stop_reason"] == "completed"


def test_agent_endpoint_rejects_short_transcript() -> None:
    response = TestClient(app).post(
        "/api/v1/agent/estimate",
        json={"transcript": "Too short"},
    )

    assert response.status_code == 422
