"""Persisted execution wrapper for the estimation graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.schemas import AgentEstimate, AgentStep, AgentTrace, GraphEstimationResult
from app.graph.build import build_estimation_graph
from app.graph.state import EstimationGraphContext


class AsyncEstimationGraph(Protocol):
    async def ainvoke(
        self,
        input: dict[str, object],
        config: dict[str, dict[str, str]],
        *,
        context: EstimationGraphContext,
    ) -> Mapping[str, object]: ...


def checkpoint_database_url(database_url: str) -> str:
    """Convert the SQLAlchemy async URL into the libpq URL psycopg expects."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def initial_graph_state(transcript: str, estimation_id: str) -> dict[str, object]:
    return {
        "estimation_id": estimation_id,
        "transcript": transcript,
        "requirements": [],
        "components": [],
        "budget_matches": [],
        "estimate": None,
        "status": None,
        "errors": [],
        "trace_steps": [],
    }


async def invoke_estimation_graph(
    graph: AsyncEstimationGraph,
    *,
    transcript: str,
    estimation_id: str,
    context: EstimationGraphContext,
) -> GraphEstimationResult:
    """Invoke one graph thread and adapt its persisted state to the API contract."""
    result = await graph.ainvoke(
        initial_graph_state(transcript, estimation_id),
        {"configurable": {"thread_id": estimation_id}},
        context=context,
    )
    estimate = AgentEstimate.model_validate(result["estimate"])
    trace = AgentTrace(steps=[AgentStep.model_validate(step) for step in result["trace_steps"]])
    return GraphEstimationResult(
        estimation_id=estimation_id,
        estimate=estimate,
        status=result["status"],
        errors=result["errors"],
        trace=trace,
        iterations=len(trace.steps),
    )


class EstimationGraphRunner:
    """Open the PostgreSQL checkpointer and run a durable graph execution."""

    def __init__(self, *, database_url: str) -> None:
        self.database_url = checkpoint_database_url(database_url)

    async def run(
        self,
        *,
        transcript: str,
        estimation_id: str,
        context: EstimationGraphContext,
    ) -> GraphEstimationResult:
        async with AsyncPostgresSaver.from_conn_string(self.database_url) as checkpointer:
            await checkpointer.setup()
            graph = build_estimation_graph(checkpointer=checkpointer)
            return await invoke_estimation_graph(
                graph,
                transcript=transcript,
                estimation_id=estimation_id,
                context=context,
            )
