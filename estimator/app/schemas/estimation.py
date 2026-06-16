from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ExampleFormat = Literal["markdown", "json", "narrative"]


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
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat


class Phase(BaseModel):
    name: str
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0)
    confidence_pct: int = Field(ge=0, le=100)
    assumptions: list[str]


class EstimationResult(BaseModel):
    summary: str
    total_duration_weeks: int = Field(ge=1)
    total_cost_eur: int = Field(ge=0)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase]

    @model_validator(mode="after")
    def total_must_match_sum_of_phases(self) -> "EstimationResult":
        sum_weeks = sum(p.duration_weeks for p in self.phases)
        sum_cost = sum(p.cost_eur for p in self.phases)
        if abs(sum_weeks - self.total_duration_weeks) > 1:
            raise ValueError("total_duration_weeks does not match sum of phase durations")
        if (
            self.total_cost_eur > 0
            and abs(sum_cost - self.total_cost_eur) / self.total_cost_eur > 0.05
        ):
            raise ValueError("total_cost_eur does not match sum of phase costs")
        return self


class EstimationResponse(BaseModel):
    result: EstimationResult
    prompt_version: str


class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]
