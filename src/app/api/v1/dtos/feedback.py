"""Feedback request and response schemas for Langfuse trace scoring."""

from typing import Optional

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request model for submitting user feedback on an agent trace."""

    trace_id: str = Field(..., description="Langfuse trace ID from the agent response")
    score: float = Field(..., ge=-1, le=1, description="Feedback score")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool
    message: str
