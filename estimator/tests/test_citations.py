"""Citation integrity: the schema rules and dangling-citation detection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.retrieval.citations import verify_citations
from app.retrieval.context import render_context
from app.retrieval.hybrid import RetrievedChunk
from app.schemas.estimation import EstimateLineItem, GroundedEstimate, SourceReference


def _source(chunk_id: str = "8") -> SourceReference:
    return SourceReference(chunk_id=chunk_id, document_id="3", evidence="Estimated hours: 120.0")


def _grounded_line(component: str = "Pagos", chunk_id: str = "8") -> EstimateLineItem:
    return EstimateLineItem(
        component=component,
        hours=120,
        rationale="Derivado del presupuesto histórico.",
        grounded=True,
        sources=[_source(chunk_id)],
    )


def _ungrounded_line(component: str = "Realidad aumentada") -> EstimateLineItem:
    return EstimateLineItem(
        component=component,
        hours=0,
        rationale="Sin datos suficientes en el contexto recuperado.",
        grounded=False,
        sources=[],
    )


def _estimate(*lines: EstimateLineItem) -> GroundedEstimate:
    return GroundedEstimate(
        summary="Estimación basada en presupuestos históricos.",
        line_items=list(lines),
        total_hours=sum(line.hours for line in lines),
    )


class TestLineIntegrity:
    def test_grounded_line_requires_a_source(self):
        with pytest.raises(ValidationError, match="cites no source"):
            EstimateLineItem(
                component="Pagos", hours=10, rationale="...", grounded=True, sources=[]
            )

    def test_ungrounded_line_cannot_invent_hours(self):
        with pytest.raises(ValidationError, match="report 0 hours for insufficient context"):
            EstimateLineItem(
                component="Pagos", hours=40, rationale="...", grounded=False, sources=[]
            )

    def test_ungrounded_line_cannot_cite(self):
        with pytest.raises(ValidationError, match="an ungrounded line has no support"):
            EstimateLineItem(
                component="Pagos", hours=0, rationale="...", grounded=False, sources=[_source()]
            )

    def test_total_must_match_line_sum(self):
        with pytest.raises(ValidationError, match="does not|adjust either the lines"):
            GroundedEstimate(
                summary="Estimación de prueba.", line_items=[_grounded_line()], total_hours=999
            )


class TestVerifyCitations:
    def test_real_citation_is_grounded(self):
        report = verify_citations(_estimate(_grounded_line()), {"8", "9"})
        assert report.is_valid
        assert len(report.grounded) == 1
        assert report.grounding_rate == 1.0

    def test_dangling_citation_is_detected(self):
        """The acceptance criterion: an id never retrieved must be caught."""
        report = verify_citations(_estimate(_grounded_line(chunk_id="9999")), {"8", "9"})

        assert not report.is_valid
        assert len(report.dangling) == 1
        assert report.dangling[0].dangling_chunk_ids == ["9999"]

    def test_insufficient_context_is_its_own_bucket(self):
        report = verify_citations(_estimate(_grounded_line(), _ungrounded_line()), {"8"})

        assert report.is_valid  # an honest gap is not a citation failure
        assert len(report.grounded) == 1
        assert len(report.insufficient) == 1
        assert report.grounding_rate == 0.5

    def test_report_separates_all_three_buckets(self):
        report = verify_citations(
            _estimate(
                _grounded_line("Pagos", "8"),
                _grounded_line("Auth", "4242"),
                _ungrounded_line(),
            ),
            {"8"},
        )
        assert report.as_dict() == {
            "lines": 3,
            "grounded": 1,
            "dangling": 1,
            "insufficient_context": 1,
            "grounding_rate": 0.333,
            "is_valid": False,
        }


class TestContextIds:
    def test_rendered_context_exposes_the_ids_it_contains(self):
        """The prompt's ids and the verifier's id set must be the same set."""
        chunks = [
            RetrievedChunk(chunk_id=8, document_id=3, chunk_type="c", content="Pagos", metadata={}),
            RetrievedChunk(
                chunk_id=9, document_id=3, chunk_type="c", content="Carrito", metadata={}
            ),
        ]
        context = render_context(chunks)

        assert context.chunk_ids == {"8", "9"}
        assert "chunk_id: 8" in context.text
        assert context.contexts == ["Pagos", "Carrito"]
