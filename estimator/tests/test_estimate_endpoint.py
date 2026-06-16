from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.routers import estimations
from app.schemas.estimation import EstimationResult, Phase


def _fake_result() -> EstimationResult:
    return EstimationResult(
        summary="A small CRM with authentication, contacts and roles.",
        total_duration_weeks=6,
        total_cost_eur=15000,
        confidence_pct=80,
        phases=[
            Phase(
                name="Discovery",
                duration_weeks=1,
                cost_eur=2500,
                confidence_pct=90,
                assumptions=["Requirements are stable"],
            ),
            Phase(
                name="Development",
                duration_weeks=4,
                cost_eur=10000,
                confidence_pct=75,
                assumptions=["No third-party integrations"],
            ),
            Phase(
                name="Deployment",
                duration_weeks=1,
                cost_eur=2500,
                confidence_pct=85,
                assumptions=["Single environment"],
            ),
        ],
    )


@pytest.fixture
def call_log(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[EstimationResult]]:
    """Replace the LLM seam with a recording fake."""
    calls: list[EstimationResult] = []

    def fake(request) -> EstimationResult:
        calls.append(request)
        return _fake_result()

    monkeypatch.setattr(estimations, "generate_estimation", fake)
    yield calls


def test_estimate_returns_structured_result(client: TestClient, call_log: list) -> None:
    payload = {
        "description": "We need a small CRM with auth, contacts and roles for six weeks.",
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "result" in body
    assert body["prompt_version"] == "v1"

    result = body["result"]
    assert result["summary"]
    assert result["total_duration_weeks"] == 6
    assert result["total_cost_eur"] == 15000
    assert len(result["phases"]) == 3
    assert result["phases"][0]["name"] == "Discovery"

    assert len(call_log) == 1
    assert call_log[0].description == payload["description"]
