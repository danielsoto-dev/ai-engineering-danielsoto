"""Responses API tool schemas and their Python implementations."""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from openai import OpenAI

from app.agent.schemas import (
    CalculateEstimateArgs,
    HistoricalBudgetItem,
    SearchBudgetsArgs,
)
from app.config import get_settings
from app.db.session import get_sessionmaker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.retrieval.context import assemble_context

log = structlog.get_logger()

CONTINGENCY_FACTOR = 0.15
SEARCH_TOP_K = 5
FILTERED_CANDIDATE_K = 10
CONTENT_PREVIEW_CHARS = 180


SEARCH_BUDGETS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_budgets",
    "description": (
        "Search historical software budgets for work analogous to one project component. "
        "Call this separately for every distinct component that needs an estimate. Use a "
        "focused query rather than the full project transcript. Results include recorded "
        "engineer-hours. Pass those values to calculate_estimate as reference_amounts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused description of one component or requirement.",
            },
            "filters": {
                "type": ["object", "null"],
                "description": "Optional search hints. Pass null when no filter is needed.",
                "properties": {
                    "sectors": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Allowed client sectors, such as logistics or industrial.",
                    },
                    "component_type": {
                        "type": ["string", "null"],
                        "description": "Component category hint, such as mobile app or ERP integration.",
                    },
                    "date_range": {
                        "type": ["object", "null"],
                        "properties": {
                            "from_year": {"type": ["integer", "null"]},
                            "to_year": {"type": ["integer", "null"]},
                        },
                        "required": ["from_year", "to_year"],
                        "additionalProperties": False,
                    },
                },
                "required": ["sectors", "component_type", "date_range"],
                "additionalProperties": False,
            },
        },
        "required": ["query", "filters"],
        "additionalProperties": False,
    },
    "strict": True,
}

CALCULATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "calculate_estimate",
    "description": (
        "Deterministically calculate engineer-hours for all identified components. For each "
        "component it takes the median historical reference amount and adds a transparent 15 "
        "percent contingency. Call this only after search_budgets has been called for every "
        "component. It returns a component breakdown and total without calling an LLM."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "reference_amounts": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Engineer-hours returned by search_budgets.",
                        },
                    },
                    "required": ["name", "reference_amounts"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["components"],
        "additionalProperties": False,
    },
    "strict": True,
}

TOOL_SCHEMAS = [SEARCH_BUDGETS_TOOL, CALCULATE_ESTIMATE_TOOL]

RetrievalBackend = Callable[[SearchBudgetsArgs], Awaitable[list[HistoricalBudgetItem]]]


def _matches_filters(item: HistoricalBudgetItem, args: SearchBudgetsArgs) -> bool:
    filters = args.filters
    if filters is None:
        return True
    if filters.sectors and (item.sector or "").lower() not in {
        sector.lower() for sector in filters.sectors
    }:
        return False
    if filters.date_range:
        if filters.date_range.from_year and (
            item.year is None or item.year < filters.date_range.from_year
        ):
            return False
        if filters.date_range.to_year and (
            item.year is None or item.year > filters.date_range.to_year
        ):
            return False
    return True


async def retrieve_historical_budgets(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
    """Adapt the existing hybrid retrieval and reranking pipeline for the agent."""
    query = args.query
    if args.filters and args.filters.component_type:
        query = f"{query}. Component type: {args.filters.component_type}"

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for budget retrieval")
    embedder = OpenAIEmbedder(client=OpenAI(api_key=settings.OPENAI_API_KEY))
    query_vector = await asyncio.to_thread(embedder.embed_one, query)
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        context = await assemble_context(
            session,
            query,
            query_vector,
            top_k=FILTERED_CANDIDATE_K,
            use_rerank=True,
        )

    items: list[HistoricalBudgetItem] = []
    for chunk in context.chunks:
        metadata = chunk.metadata
        hours = metadata.get("estimated_hours")
        if hours is None:
            continue
        item = HistoricalBudgetItem(
            id=chunk.chunk_id,
            content_preview=" ".join(chunk.content.split())[:CONTENT_PREVIEW_CHARS],
            sector=metadata.get("client_sector"),
            budget_id=metadata.get("budget_id"),
            estimated_hours=float(hours),
            year=metadata.get("year"),
            distance=chunk.vector_distance,
            rerank_score=chunk.rerank_score,
        )
        if _matches_filters(item, args):
            items.append(item)
        if len(items) == SEARCH_TOP_K:
            break
    return items


async def search_budgets(
    raw_arguments: dict[str, Any],
    *,
    backend: RetrievalBackend = retrieve_historical_budgets,
) -> dict[str, Any]:
    args = SearchBudgetsArgs.model_validate(raw_arguments)
    items = await backend(args)
    hours = [item.estimated_hours for item in items]
    log.info("agent_search_budgets", query=args.query, results=len(items), hours=hours)
    return {
        "items": [item.model_dump() for item in items],
        "count": len(items),
        "summary": f"found {len(items)} references for {args.query!r}; hours={hours}",
    }


def calculate_estimate(raw_arguments: dict[str, Any]) -> dict[str, Any]:
    args = CalculateEstimateArgs.model_validate(raw_arguments)
    breakdown: list[dict[str, Any]] = []
    total_hours = 0.0

    for component in args.components:
        if component.reference_amounts:
            median_hours = statistics.median(component.reference_amounts)
            estimated_hours = round(median_hours * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            estimated_hours = 0.0
            unbudgeted = True
        total_hours += estimated_hours
        breakdown.append(
            {
                "name": component.name,
                "reference_count": len(component.reference_amounts),
                "estimated_hours": estimated_hours,
                "unbudgeted": unbudgeted,
            }
        )

    total_hours = round(total_hours, 1)
    log.info("agent_calculate_estimate", components=len(breakdown), total_hours=total_hours)
    return {
        "components": breakdown,
        "total_hours": total_hours,
        "contingency_factor": CONTINGENCY_FACTOR,
        "summary": f"calculated {total_hours}h across {len(breakdown)} components",
    }


async def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    retrieval_backend: RetrievalBackend = retrieve_historical_budgets,
) -> dict[str, Any]:
    if name == "search_budgets":
        return await search_budgets(arguments, backend=retrieval_backend)
    if name == "calculate_estimate":
        return calculate_estimate(arguments)
    raise ValueError(f"Unknown tool: {name}")
