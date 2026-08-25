"""Post-generation citation verification.

A citation the model invented is worse than no citation: it carries the same
authority and none of the backing. This module compares every cited chunk_id
against the set actually handed to the LLM and reports the ones that were never
there, so a dangling citation surfaces as a quality failure instead of passing
for rigour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from app.schemas.estimation import GroundedEstimate

log = structlog.get_logger()


@dataclass
class LineVerdict:
    component: str
    status: str  # "grounded" | "dangling" | "insufficient_context"
    cited_chunk_ids: list[str] = field(default_factory=list)
    dangling_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class CitationReport:
    """Per-line verdicts plus the roll-up the caller logs and returns."""

    verdicts: list[LineVerdict]

    @property
    def grounded(self) -> list[LineVerdict]:
        return [v for v in self.verdicts if v.status == "grounded"]

    @property
    def dangling(self) -> list[LineVerdict]:
        return [v for v in self.verdicts if v.status == "dangling"]

    @property
    def insufficient(self) -> list[LineVerdict]:
        return [v for v in self.verdicts if v.status == "insufficient_context"]

    @property
    def is_valid(self) -> bool:
        """True when no line cites a chunk outside the retrieved context."""
        return not self.dangling

    @property
    def grounding_rate(self) -> float:
        """Share of lines that are grounded in a real source."""
        if not self.verdicts:
            return 0.0
        return len(self.grounded) / len(self.verdicts)

    def as_dict(self) -> dict:
        return {
            "lines": len(self.verdicts),
            "grounded": len(self.grounded),
            "dangling": len(self.dangling),
            "insufficient_context": len(self.insufficient),
            "grounding_rate": round(self.grounding_rate, 3),
            "is_valid": self.is_valid,
        }


def verify_citations(
    estimate: GroundedEstimate,
    retrieved_chunk_ids: set[str],
    *,
    request_id: str | None = None,
) -> CitationReport:
    """Flag any line whose cited chunk_id was never in the retrieved context."""
    verdicts: list[LineVerdict] = []

    for item in estimate.line_items:
        if not item.grounded:
            verdicts.append(LineVerdict(component=item.component, status="insufficient_context"))
            continue

        cited = [source.chunk_id for source in item.sources]
        dangling = [chunk_id for chunk_id in cited if chunk_id not in retrieved_chunk_ids]
        verdicts.append(
            LineVerdict(
                component=item.component,
                status="dangling" if dangling else "grounded",
                cited_chunk_ids=cited,
                dangling_chunk_ids=dangling,
            )
        )

    report = CitationReport(verdicts=verdicts)

    log_event = log.bind(request_id=request_id) if request_id else log
    if report.dangling:
        log_event.error(
            "citation_verification_failed",
            **report.as_dict(),
            dangling_lines=[
                {"component": v.component, "dangling_chunk_ids": v.dangling_chunk_ids}
                for v in report.dangling
            ],
        )
    else:
        log_event.info("citation_verification_passed", **report.as_dict())

    return report
