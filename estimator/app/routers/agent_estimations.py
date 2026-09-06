"""HTTP entry point for the Session 12 estimation agent."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.agent.loop import run_estimation_agent
from app.agent.schemas import AgentRunResult
from app.config import get_settings
from app.dependencies import get_agent_openai_client
from app.guardrails.input import InputGuardrailViolation, check_input

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentEstimationRequest(BaseModel):
    transcript: str = Field(min_length=20, max_length=80_000)


@router.post("/estimate", response_model=AgentRunResult)
async def create_agent_estimation(
    payload: AgentEstimationRequest,
    client: AsyncOpenAI | None = Depends(get_agent_openai_client),
) -> AgentRunResult:
    """Run the manual reason, act, observe loop for one transcript."""
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is required")

    settings = get_settings()
    try:
        check_input(payload.transcript)
        return await run_estimation_agent(
            payload.transcript,
            client=client,
            model=settings.AGENT_MODEL,
            reasoning_effort=settings.AGENT_REASONING_EFFORT,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
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
