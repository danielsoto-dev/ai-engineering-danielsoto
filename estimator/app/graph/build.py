"""Build the sequential Session 13 estimation graph."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    classify_components,
    extract_requirements,
    generate_estimate,
    search_component_budgets,
    validate_and_consolidate,
)
from app.graph.state import EstimationGraphContext, EstimationState


def route_after_validation(
    state: EstimationState,
) -> Literal["validated", "needs_review"]:
    return state["status"] or "needs_review"


def build_estimation_graph(checkpointer=None):
    builder = StateGraph(EstimationState, context_schema=EstimationGraphContext)
    builder.add_node("extract_requirements", extract_requirements)
    builder.add_node("classify_components", classify_components)
    builder.add_node("search_budgets", search_component_budgets)
    builder.add_node("generate_estimate", generate_estimate)
    builder.add_node("validate_and_consolidate", validate_and_consolidate)

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    builder.add_conditional_edges(
        "validate_and_consolidate",
        route_after_validation,
        {"validated": END, "needs_review": END},
    )
    return builder.compile(checkpointer=checkpointer)
