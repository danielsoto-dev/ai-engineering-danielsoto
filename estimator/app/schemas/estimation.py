"""Request and response models for the estimation endpoint.

Session 4 contract: typed form-style request maps to a typed, validated
``EstimationResult`` (structured output via Instructor + Pydantic). Two model
validators enforce business rules that the LLM cannot break:

1. The cost of all phases must sum to ``total_cost_eur``.
2. Low-confidence answers (< 30%) must declare it explicitly by starting the
   summary with ``"Out of scope:"``.

When the LLM violates a validator, Instructor re-prompts the model with the
``ValueError`` message until it agrees (up to ``max_retries`` attempts).
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    """Typed payload sent by the business backend or Streamlit form."""

    description: str = Field(
        min_length=20,
        max_length=80000,
        description="Free-text description or transcription of the project to estimate.",
    )
    project_type: ProjectType = Field(description="Coarse-grained project category.")
    detail_level: DetailLevel = Field(description="How deep the estimation should go.")
    output_format: OutputFormat = Field(description="Shape of the rendered estimation.")


# --- Structured response ----------------------------------------------------


OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """One phase in the breakdown of an estimation."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    """Structured estimation. The two validators below are the business rules
    that the LLM cannot break — Instructor will re-prompt the model when one
    of them raises.

    Field order is deliberate: ``phases`` comes BEFORE the totals so the LLM
    commits to the per-phase numbers first (autoregressive generation) and
    then only needs to sum them when filling the totals. Putting totals first
    leads the model to pick a round number and then back-fit phases to it,
    which it does very badly arithmetically — particularly with smaller
    models like ``gpt-4o-mini``.
    """

    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> "EstimationResult":
        phase_sum = sum(p.cost_eur for p in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(
                f"phases sum ({phase_sum} EUR) does not match total_cost_eur "
                f"({self.total_cost_eur} EUR); adjust either the phases or the total"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self


class EstimationResponse(BaseModel):
    """Wraps the validated result, the prompt version that produced it, and
    whether it came from a cache (exact or semantic)."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False


from app.schemas.acb import BossTrace  # noqa: E402


class ACBResponse(EstimationResponse):
    """Conversational response with the Actor-Critic-Boss audit trail.

    Same shape as ``EstimationResponse`` plus the ``acb`` field carrying the
    iteration log. The UI uses the trail to render an expander showing what
    the Critic flagged at each step and how the Boss decided.
    """

    acb: BossTrace


# --- Per-line grounded citation (Session 11) --------------------------------


class SourceReference(BaseModel):
    """One retrieved chunk backing one estimation line."""

    chunk_id: str = Field(
        min_length=1, description="Id of the retrieved chunk supporting the line."
    )
    document_id: str = Field(
        min_length=1, description="Historical budget document the chunk belongs to."
    )
    evidence: str = Field(
        min_length=1,
        max_length=600,
        description="Verbatim span or figure copied from the source, never a paraphrase.",
    )


class EstimateLineItem(BaseModel):
    """A single component of the estimation, attributed to its sources.

    ``grounded`` is the honesty switch. A line the model can support must carry
    at least one source; a line it cannot must report zero hours rather than
    invent a plausible number, which is the failure mode this schema exists to
    make impossible.
    """

    component: str = Field(min_length=1, max_length=120)
    hours: float = Field(ge=0, le=10_000)
    rationale: str = Field(min_length=1, max_length=800)
    grounded: bool = Field(
        description="True only when the line is derived from the retrieved context."
    )
    sources: list[SourceReference] = Field(
        default_factory=list, description="Non-empty if and only if grounded is True."
    )

    @model_validator(mode="after")
    def sources_match_grounded(self) -> "EstimateLineItem":
        if self.grounded and not self.sources:
            raise ValueError(
                f"line {self.component!r} is marked grounded=True but cites no source; "
                f"cite at least one chunk_id from the context or set grounded=False"
            )
        if not self.grounded:
            if self.sources:
                raise ValueError(
                    f"line {self.component!r} is marked grounded=False but cites "
                    f"{len(self.sources)} source(s); an ungrounded line has no support"
                )
            if self.hours != 0:
                raise ValueError(
                    f"line {self.component!r} is marked grounded=False but claims "
                    f"{self.hours} hours; report 0 hours for insufficient context "
                    f"instead of estimating without support"
                )
        return self


class GroundedEstimate(BaseModel):
    """Estimation whose every line is individually attributable."""

    summary: str = Field(min_length=10, max_length=1200)
    line_items: list[EstimateLineItem] = Field(min_length=1, max_length=20)
    total_hours: float = Field(ge=0, le=100_000)

    @model_validator(mode="after")
    def total_matches_grounded_lines(self) -> "GroundedEstimate":
        line_sum = sum(item.hours for item in self.line_items)
        if abs(line_sum - self.total_hours) > 0.01:
            raise ValueError(
                f"line items sum to {line_sum} hours but total_hours is "
                f"{self.total_hours}; adjust either the lines or the total"
            )
        return self

    def as_text(self) -> str:
        """Flatten to prose. RAGAS scores an ``answer`` string, not an object."""
        lines = [self.summary, ""]
        for item in self.line_items:
            if item.grounded:
                cited = ", ".join(source.chunk_id for source in item.sources)
                lines.append(
                    f"- {item.component}: {item.hours:g} h. {item.rationale} [fuentes: {cited}]"
                )
            else:
                lines.append(f"- {item.component}: sin datos suficientes. {item.rationale}")
        lines.append("")
        lines.append(f"Total: {self.total_hours:g} horas.")
        return "\n".join(lines)


class GroundedEstimationRequest(BaseModel):
    """Free-text project description to estimate against historical budgets."""

    description: str = Field(
        min_length=20,
        max_length=80000,
        description="Description or transcription of the project to estimate.",
    )


class CitedLine(BaseModel):
    """Per-line verification verdict returned alongside the estimate."""

    component: str
    status: str = Field(description="grounded | dangling | insufficient_context")
    cited_chunk_ids: list[str] = Field(default_factory=list)
    dangling_chunk_ids: list[str] = Field(default_factory=list)


class CitationSummary(BaseModel):
    lines: int
    grounded: int
    dangling: int
    insufficient_context: int
    grounding_rate: float
    is_valid: bool = Field(description="False when any line cites a chunk never retrieved.")


class GroundedEstimationResponse(BaseModel):
    """The estimate, its per-line verification, and the context it was given.

    The HTTP contract is unchanged in shape — the body is enriched with the
    sources behind each line so a consumer can audit the estimate instead of
    trusting it.
    """

    estimate: GroundedEstimate
    citations: CitationSummary
    verified_lines: list[CitedLine]
    retrieved_chunk_ids: list[str]
