"""HTTP tests for the Session 12 agent endpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.dependencies import get_agent_openai_client
from app.main import app


class _FakeResponses:
    async def create(self, **kwargs):
        estimate = {
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
        return SimpleNamespace(
            id="response_1",
            output=[SimpleNamespace(type="message")],
            output_text=json.dumps(estimate),
        )


class _FakeClient:
    responses = _FakeResponses()


def test_agent_endpoint_returns_estimate_and_trace() -> None:
    app.dependency_overrides[get_agent_openai_client] = lambda: _FakeClient()
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
    assert body["stop_reason"] == "completed"


def test_agent_endpoint_rejects_short_transcript() -> None:
    response = TestClient(app).post(
        "/api/v1/agent/estimate",
        json={"transcript": "Too short"},
    )

    assert response.status_code == 422
