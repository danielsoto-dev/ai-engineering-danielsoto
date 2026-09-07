"""Typed shared state and runtime context for the estimation graph."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from openai import AsyncOpenAI

from app.agent.tools import RetrievalBackend, retrieve_historical_budgets


class Component(TypedDict):
    name: str
    category: str
    search_query: str


class BudgetMatch(TypedDict):
    component: str
    reference_budget_id: str
    chunk_id: int
    amount: float
    content_preview: str


class GraphTraceStep(TypedDict):
    step: int
    reasoning: str
    action: str
    arguments: dict[str, object]
    observation: str


class EstimationState(TypedDict):
    estimation_id: str
    transcript: str
    requirements: list[str]
    components: list[Component]
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    estimate: dict[str, object] | None
    status: Literal["validated", "needs_review"] | None
    errors: Annotated[list[str], operator.add]
    trace_steps: Annotated[list[GraphTraceStep], operator.add]


@dataclass(frozen=True)
class EstimationGraphContext:
    client: AsyncOpenAI
    model: str = "gpt-5"
    reasoning_effort: str = "medium"
    retrieval_backend: RetrievalBackend = retrieve_historical_budgets
