"""Estimation orchestration: prompt building via Jinja2 templates and structured
output generation through Instructor.

The module exposes a single entrypoint, :func:`generate_estimation`, which takes
an :class:`app.schemas.estimation.EstimationRequest` and returns a validated
:class:`app.schemas.estimation.EstimationResult`. All provider-specific details
are handled by Instructor + OpenAI structured outputs.
"""

from __future__ import annotations

from functools import lru_cache

import instructor
import structlog
from openai import OpenAI

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResult

log = structlog.get_logger()


@lru_cache(maxsize=1)
def _get_client():
    """Lazy Instructor client so tests can import without an API key."""
    return instructor.from_openai(OpenAI())


def generate_estimation(request: EstimationRequest) -> EstimationResult:
    """Generate a structured software estimation from the user request."""
    system_prompt, user_prompt = render_estimation_prompt(request)

    log.info(
        "generating_estimation",
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
    )

    result: EstimationResult = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        response_model=EstimationResult,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    log.info(
        "estimation_completed",
        total_cost_eur=result.total_cost_eur,
        total_duration_weeks=result.total_duration_weeks,
        phase_count=len(result.phases),
    )

    return result
