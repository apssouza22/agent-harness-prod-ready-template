"""Checkpoint API schemas for chatbot state and history endpoints."""

from pydantic import BaseModel, Field

from src.app.core.checkpoint.models import CheckpointDetail, CheckpointSummary, StateHistoryEntry


class CheckpointListResponse(BaseModel):
    """Response model for listing session checkpoints."""

    checkpoints: list[CheckpointSummary] = Field(default_factory=list, description="Checkpoint summaries")


class CheckpointDetailResponse(BaseModel):
    """Response model for a single checkpoint."""

    checkpoint: CheckpointDetail = Field(..., description="Checkpoint detail")


class StateHistoryResponse(BaseModel):
    """Response model for graph state history."""

    history: list[StateHistoryEntry] = Field(default_factory=list, description="State history snapshots")
