"""Validated tool arguments, trace records, and final agent output."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DateRange(BaseModel):
    """Optional year range for historical budget searches."""

    from_year: int | None = None
    to_year: int | None = None

    @model_validator(mode="after")
    def validate_order(self) -> "DateRange":
        if self.from_year and self.to_year and self.from_year > self.to_year:
            raise ValueError("from_year must be less than or equal to to_year")
        return self


class SearchFilters(BaseModel):
    """Optional metadata hints accepted by ``search_budgets``."""

    sectors: list[str] | None = None
    component_type: str | None = None
    date_range: DateRange | None = None


class SearchBudgetsArgs(BaseModel):
    query: str = Field(min_length=1)
    filters: SearchFilters | None = None


class EstimateComponentInput(BaseModel):
    name: str = Field(min_length=1)
    reference_amounts: list[float]


class CalculateEstimateArgs(BaseModel):
    components: list[EstimateComponentInput] = Field(min_length=1)


class HistoricalBudgetItem(BaseModel):
    id: int
    content_preview: str
    sector: str | None = None
    budget_id: str | None = None
    estimated_hours: float
    year: int | None = None
    distance: float | None = None
    rerank_score: float | None = None


class AgentComponentEstimate(BaseModel):
    name: str
    estimated_hours: float = Field(ge=0)
    reference_chunk_ids: list[int] = Field(default_factory=list)
    rationale: str


class AgentEstimate(BaseModel):
    components: list[AgentComponentEstimate]
    total_hours: float = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class AgentStep(BaseModel):
    step: int = Field(ge=1)
    reasoning: str
    action: str
    arguments: dict[str, Any]
    observation: str


class AgentTrace(BaseModel):
    steps: list[AgentStep] = Field(default_factory=list)

    def render(self) -> str:
        blocks: list[str] = []
        for step in self.steps:
            arguments = json.dumps(step.arguments, ensure_ascii=False)
            blocks.append(
                f"STEP {step.step}\n"
                f"  reasoning: {step.reasoning}\n"
                f"  action: {step.action}({arguments})\n"
                f"  observation: {step.observation}"
            )
        return "\n\n".join(blocks) if blocks else "(no tool calls)"


class AgentRunResult(BaseModel):
    estimate: AgentEstimate | None
    trace: AgentTrace
    iterations: int = Field(ge=1)
    stop_reason: Literal["completed", "max_iterations", "invalid_final_output"]


class GraphEstimationResult(BaseModel):
    estimation_id: str
    estimate: AgentEstimate
    status: Literal["validated", "needs_review"]
    errors: list[str] = Field(default_factory=list)
    trace: AgentTrace
    iterations: int = Field(ge=1)
    stop_reason: Literal["completed"] = "completed"
