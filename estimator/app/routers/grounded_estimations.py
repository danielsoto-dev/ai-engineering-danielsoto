"""POST /api/v1/estimate/grounded — estimation with auditable per-line sources.

Same error mapping as the plain estimate endpoint. A dangling citation does not
fail the request: the estimate is returned with ``citations.is_valid`` false and
the offending lines marked, so the caller decides whether to trust it. Hiding a
failed verification behind a 502 would lose the very information the check
exists to surface.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.dependencies import get_grounded_estimation_service
from app.guardrails.input import InputGuardrailViolation, check_input
from app.schemas.estimation import (
    CitationSummary,
    CitedLine,
    GroundedEstimationRequest,
    GroundedEstimationResponse,
)
from app.services.grounded_estimation import GroundedEstimationService

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])


@router.post("/estimate/grounded", response_model=GroundedEstimationResponse)
async def create_grounded_estimation(
    payload: GroundedEstimationRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    service: GroundedEstimationService = Depends(get_grounded_estimation_service),
) -> GroundedEstimationResponse:
    """Retrieve, generate with per-line attribution, and verify the citations."""
    request_id = request.headers.get("x-request-id")

    log.info(
        "grounded_estimation_request_received",
        request_id=request_id,
        description_chars=len(payload.description),
    )

    try:
        check_input(payload.description)
        outcome = await service.estimate(session, payload.description, request_id=request_id)
    except InputGuardrailViolation as exc:
        log.info(
            "grounded_estimation_blocked_by_input_guardrail",
            request_id=request_id,
            reason=exc.reason,
        )
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except Exception as exc:
        log.error(
            "grounded_estimation_endpoint_error",
            request_id=request_id,
            error=str(exc)[:400],
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc

    return GroundedEstimationResponse(
        estimate=outcome.estimate,
        citations=CitationSummary(**outcome.citations.as_dict()),
        verified_lines=[
            CitedLine(
                component=verdict.component,
                status=verdict.status,
                cited_chunk_ids=verdict.cited_chunk_ids,
                dangling_chunk_ids=verdict.dangling_chunk_ids,
            )
            for verdict in outcome.citations.verdicts
        ],
        retrieved_chunk_ids=outcome.context.sorted_chunk_ids,
    )
