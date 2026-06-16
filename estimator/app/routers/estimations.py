import structlog
from fastapi import APIRouter

from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import generate_estimation

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(request: EstimationRequest) -> EstimationResponse:
    """Receive a project description and return a structured estimation."""
    log.info("estimate_request_received", project_type=request.project_type.value)

    result = generate_estimation(request)

    return EstimationResponse(
        result=result,
        prompt_version="v1",
    )
