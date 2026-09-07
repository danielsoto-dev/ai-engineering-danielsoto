"""Sequential nodes for the estimation graph."""

from __future__ import annotations

import json
from collections import defaultdict

import logfire
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agent.schemas import AgentComponentEstimate, AgentEstimate, SearchBudgetsArgs
from app.agent.tools import calculate_estimate, search_budgets
from app.graph.state import (
    BudgetMatch,
    Component,
    EstimationGraphContext,
    EstimationState,
    GraphTraceStep,
)


class _RequirementExtraction(BaseModel):
    requirements: list[str] = Field(min_length=1)


class _ComponentPayload(BaseModel):
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    search_query: str = Field(min_length=1)


class _ComponentClassification(BaseModel):
    components: list[_ComponentPayload] = Field(min_length=1)


REQUIREMENTS_FORMAT = {
    "type": "json_schema",
    "name": "requirement_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    },
}

COMPONENTS_FORMAT = {
    "type": "json_schema",
    "name": "component_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                        "search_query": {"type": "string"},
                    },
                    "required": ["name", "category", "search_query"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["components"],
        "additionalProperties": False,
    },
}


def _trace_step(
    state: EstimationState,
    *,
    action: str,
    reasoning: str,
    arguments: dict[str, object],
    observation: str,
) -> GraphTraceStep:
    return {
        "step": len(state["trace_steps"]) + 1,
        "reasoning": reasoning,
        "action": action,
        "arguments": arguments,
        "observation": observation,
    }


async def extract_requirements(
    state: EstimationState,
    runtime: Runtime[EstimationGraphContext],
) -> dict[str, object]:
    """Convert the transcript into a complete list of requirements."""
    with logfire.span("node: extract_requirements", estimation_id=state["estimation_id"]):
        response = await runtime.context.client.responses.create(
            model=runtime.context.model,
            instructions=(
                "Extract every functional and non-functional software requirement from the "
                "transcript. Keep distinct capabilities separate. Do not estimate effort."
            ),
            input=state["transcript"],
            reasoning={"effort": runtime.context.reasoning_effort, "summary": "auto"},
            text={"format": REQUIREMENTS_FORMAT},
            store=True,
        )
        extracted = _RequirementExtraction.model_validate_json(response.output_text)
        requirements = [item.strip() for item in extracted.requirements if item.strip()]
        return {
            "requirements": requirements,
            "trace_steps": [
                _trace_step(
                    state,
                    action="extract_requirements",
                    reasoning="Turn the transcript into explicit requirements before grouping work.",
                    arguments={"transcript_chars": len(state["transcript"])},
                    observation=f"extracted {len(requirements)} requirements",
                )
            ],
        }


async def classify_components(
    state: EstimationState,
    runtime: Runtime[EstimationGraphContext],
) -> dict[str, object]:
    """Group requirements into independently estimable components."""
    with logfire.span("node: classify_components", estimation_id=state["estimation_id"]):
        response = await runtime.context.client.responses.create(
            model=runtime.context.model,
            instructions=(
                "Group the requirements into distinct software components. Give each component "
                "a short name, a broad category, and one focused historical-budget search query. "
                "Do not combine unrelated work."
            ),
            input=json.dumps(state["requirements"], ensure_ascii=False),
            reasoning={"effort": runtime.context.reasoning_effort, "summary": "auto"},
            text={"format": COMPONENTS_FORMAT},
            store=True,
        )
        classified = _ComponentClassification.model_validate_json(response.output_text)
        components: list[Component] = [
            {
                "name": item.name,
                "category": item.category,
                "search_query": item.search_query,
            }
            for item in classified.components
        ]
        return {
            "components": components,
            "trace_steps": [
                _trace_step(
                    state,
                    action="classify_components",
                    reasoning="Group related requirements without hiding separate areas of work.",
                    arguments={"requirements": len(state["requirements"])},
                    observation=f"classified {len(components)} components",
                )
            ],
        }


async def search_component_budgets(
    state: EstimationState,
    runtime: Runtime[EstimationGraphContext],
) -> dict[str, object]:
    """Search historical budgets for every component, one after another."""
    with logfire.span("node: search_budgets", estimation_id=state["estimation_id"]):
        matches: list[BudgetMatch] = []
        errors: list[str] = []
        result_counts: dict[str, object] = {}
        for component in state["components"]:
            result = await search_budgets(
                SearchBudgetsArgs(
                    query=component["search_query"],
                    filters=None,
                ).model_dump(),
                backend=runtime.context.retrieval_backend,
            )
            items = result["items"]
            result_counts[component["name"]] = len(items)
            if not items:
                errors.append(f"No historical budgets found for {component['name']}.")
            for item in items:
                matches.append(
                    {
                        "component": component["name"],
                        "reference_budget_id": item.get("budget_id") or f"chunk-{item['id']}",
                        "chunk_id": item["id"],
                        "amount": item["estimated_hours"],
                        "content_preview": item["content_preview"],
                    }
                )

        return {
            "budget_matches": matches,
            "errors": errors,
            "trace_steps": [
                _trace_step(
                    state,
                    action="search_budgets",
                    reasoning="Search each classified component sequentially against historical data.",
                    arguments={"queries": len(state["components"])},
                    observation=f"found {len(matches)} matches; per component={result_counts}",
                )
            ],
        }


def generate_estimate(state: EstimationState) -> dict[str, object]:
    """Create one deterministic estimate from the retrieved budget matches."""
    with logfire.span("node: generate_estimate", estimation_id=state["estimation_id"]):
        grouped: dict[str, list[BudgetMatch]] = defaultdict(list)
        for match in state["budget_matches"]:
            grouped[match["component"]].append(match)

        calculation = calculate_estimate(
            {
                "components": [
                    {
                        "name": component["name"],
                        "reference_amounts": [
                            match["amount"] for match in grouped[component["name"]]
                        ],
                    }
                    for component in state["components"]
                ]
            }
        )
        calculated_by_name = {item["name"]: item for item in calculation["components"]}
        minimum_references = min(
            (len(grouped[component["name"]]) for component in state["components"]),
            default=0,
        )
        confidence = (
            "low" if minimum_references == 0 else "high" if minimum_references >= 3 else "medium"
        )
        estimate = AgentEstimate(
            components=[
                AgentComponentEstimate(
                    name=component["name"],
                    estimated_hours=calculated_by_name[component["name"]]["estimated_hours"],
                    reference_chunk_ids=[match["chunk_id"] for match in grouped[component["name"]]],
                    rationale=(
                        f"Median of {len(grouped[component['name']])} historical references "
                        "plus 15% contingency."
                    ),
                )
                for component in state["components"]
            ],
            total_hours=calculation["total_hours"],
            assumptions=[
                "Historical engineer-hours are comparable to the requested scope.",
                "The deterministic calculator adds a 15% contingency per component.",
            ],
            confidence=confidence,
        )
        return {
            "estimate": estimate.model_dump(),
            "trace_steps": [
                _trace_step(
                    state,
                    action="generate_estimate",
                    reasoning="Use only retrieved hours and the deterministic calculator.",
                    arguments={"budget_matches": len(state["budget_matches"])},
                    observation=f"calculated {estimate.total_hours}h",
                )
            ],
        }


def validate_and_consolidate(state: EstimationState) -> dict[str, object]:
    """Validate the final structure and assign its public status."""
    with logfire.span("node: validate_and_consolidate", estimation_id=state["estimation_id"]):
        errors: list[str] = []
        estimate = AgentEstimate.model_validate(state["estimate"])
        component_names = {match["component"] for match in state["budget_matches"]}
        for component in estimate.components:
            if component.name not in component_names:
                errors.append(f"Missing historical references for {component.name}.")
        calculated_total = round(
            sum(component.estimated_hours for component in estimate.components), 1
        )
        if calculated_total != estimate.total_hours:
            errors.append(
                f"Component total {calculated_total}h does not match {estimate.total_hours}h."
            )

        status = "needs_review" if state["errors"] or errors else "validated"
        return {
            "status": status,
            "errors": errors,
            "trace_steps": [
                _trace_step(
                    state,
                    action="validate_and_consolidate",
                    reasoning="Reject totals without complete references or consistent arithmetic.",
                    arguments={"components": len(estimate.components)},
                    observation=f"status={status}; errors={len(state['errors']) + len(errors)}",
                )
            ],
        }
