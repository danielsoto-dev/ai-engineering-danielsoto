"""End-to-end tests for POST /api/v1/estimate/grounded.

The service is faked so the endpoint is exercised without network or database
access. What matters here is response shaping: the per-line sources and the
verification verdicts must reach the caller, including when a citation dangles.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db_session
from app.dependencies import get_grounded_estimation_service
from app.main import app
from app.retrieval.citations import verify_citations
from app.retrieval.context import render_context
from app.retrieval.hybrid import RetrievedChunk
from app.schemas.estimation import (
    EstimateLineItem,
    GroundedEstimate,
    SourceReference,
)
from app.services.grounded_estimation import GroundedEstimationOutcome

DESCRIPTION = "Necesitamos una tienda online con catálogo, carrito y proceso de pago."


def _context():
    return render_context(
        [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=5,
                chunk_type="budget_component",
                content=f"Component {chunk_id}\nEstimated hours: 220.0",
                metadata={},
            )
            for chunk_id in (11, 4, 8)
        ]
    )


def _estimate(cited_chunk_id: str) -> GroundedEstimate:
    return GroundedEstimate(
        summary="Estimación basada en presupuestos históricos comparables.",
        line_items=[
            EstimateLineItem(
                component="Catálogo de productos",
                hours=220,
                rationale="Componente equivalente en un presupuesto histórico.",
                grounded=True,
                sources=[
                    SourceReference(
                        chunk_id=cited_chunk_id,
                        document_id="5",
                        evidence="Estimated hours: 220.0",
                    )
                ],
            ),
            EstimateLineItem(
                component="Proceso de pago",
                hours=0,
                rationale="Sin datos suficientes en el contexto recuperado.",
                grounded=False,
                sources=[],
            ),
        ],
        total_hours=220,
    )


class _FakeService:
    def __init__(self, cited_chunk_id: str) -> None:
        self.cited_chunk_id = cited_chunk_id

    async def estimate(self, session, description, *, request_id=None):
        context = _context()
        estimate = _estimate(self.cited_chunk_id)
        return GroundedEstimationOutcome(
            estimate=estimate,
            citations=verify_citations(estimate, context.chunk_ids, request_id=request_id),
            context=context,
            meta={},
        )


@pytest.fixture
def client():
    async def _no_db():
        yield None

    app.dependency_overrides[get_db_session] = _no_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _post(client, cited_chunk_id: str):
    app.dependency_overrides[get_grounded_estimation_service] = lambda: _FakeService(cited_chunk_id)
    return client.post("/api/v1/estimate/grounded", json={"description": DESCRIPTION})


class TestGroundedEndpoint:
    def test_returns_per_line_sources(self, client):
        body = _post(client, "8").json()

        line = body["estimate"]["line_items"][0]
        assert line["grounded"] is True
        assert line["sources"][0]["chunk_id"] == "8"
        assert line["sources"][0]["evidence"] == "Estimated hours: 220.0"

    def test_reports_verification_to_the_caller(self, client):
        body = _post(client, "8").json()

        assert body["citations"] == {
            "lines": 2,
            "grounded": 1,
            "dangling": 0,
            "insufficient_context": 1,
            "grounding_rate": 0.5,
            "is_valid": True,
        }
        assert [line["status"] for line in body["verified_lines"]] == [
            "grounded",
            "insufficient_context",
        ]

    def test_dangling_citation_surfaces_without_failing_the_request(self, client):
        """A dangling citation is reported, not hidden behind a 502 — the caller
        needs the information the check exists to produce."""
        response = _post(client, "9999")

        assert response.status_code == 200
        body = response.json()
        assert body["citations"]["is_valid"] is False
        assert body["verified_lines"][0]["status"] == "dangling"
        assert body["verified_lines"][0]["dangling_chunk_ids"] == ["9999"]

    def test_chunk_ids_are_numerically_ordered(self, client):
        body = _post(client, "8").json()
        assert body["retrieved_chunk_ids"] == ["4", "8", "11"]

    def test_short_description_is_rejected(self, client):
        app.dependency_overrides[get_grounded_estimation_service] = lambda: _FakeService("8")
        response = client.post("/api/v1/estimate/grounded", json={"description": "corto"})
        assert response.status_code == 422
