"""HTTP entry point for the Session 12 estimation agent."""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.agent.schemas import GraphEstimationResult
from app.config import get_settings
from app.dependencies import get_agent_openai_client, get_estimation_graph_runner
from app.graph.runner import EstimationGraphRunner
from app.graph.state import EstimationGraphContext
from app.guardrails.input import InputGuardrailViolation, check_input

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentEstimationRequest(BaseModel):
    transcript: str = Field(min_length=20, max_length=80_000)


@router.post("/estimate", response_model=GraphEstimationResult)
async def create_agent_estimation(
    payload: AgentEstimationRequest,
    client: AsyncOpenAI | None = Depends(get_agent_openai_client),
    runner: EstimationGraphRunner = Depends(get_estimation_graph_runner),
) -> GraphEstimationResult:
    """Run one persisted LangGraph estimation."""
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is required")

    settings = get_settings()
    try:
        check_input(payload.transcript)
        estimation_id = str(uuid4())
        return await runner.run(
            transcript=payload.transcript,
            estimation_id=estimation_id,
            context=EstimationGraphContext(
                client=client,
                model=settings.AGENT_MODEL,
                reasoning_effort=settings.AGENT_REASONING_EFFORT,
            ),
        )
    except InputGuardrailViolation as exc:
        raise HTTPException(
            status_code=400,
            detail={"reason": exc.reason, "message": exc.message},
        ) from exc
    except Exception as exc:
        log.error(
            "agent_estimation_endpoint_error",
            error=str(exc)[:400],
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Agent execution failed") from exc
