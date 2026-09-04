"""Langfuse feedback API endpoint."""

from fastapi import APIRouter, HTTPException, Request

from src.app.api.security.limiter import limiter
from src.app.api.v1.dtos.feedback import FeedbackRequest, FeedbackResponse
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.dependencies import LangfuseDep

router = APIRouter()


@router.post("", response_model=FeedbackResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["feedback"][0])
async def submit_feedback(
    request: Request,
    feedback_request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """Submit user feedback linked to a Langfuse trace."""
    if not langfuse_tracer or not langfuse_tracer.client:
        raise HTTPException(status_code=503, detail="Langfuse tracing is disabled.")

    success = langfuse_tracer.submit_feedback(
        trace_id=feedback_request.trace_id,
        score=feedback_request.score,
        comment=feedback_request.comment,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

    langfuse_tracer.flush()
    logger.info("feedback_submitted", trace_id=feedback_request.trace_id, score=feedback_request.score)
    return FeedbackResponse(success=True, message="Feedback recorded successfully")
