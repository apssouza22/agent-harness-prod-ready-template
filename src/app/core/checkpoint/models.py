"""Pydantic models and serializers for LangGraph checkpoint data."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import convert_to_openai_messages
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.types import StateSnapshot
from pydantic import BaseModel, Field


class CheckpointSummary(BaseModel):
    """Summary metadata for a single LangGraph checkpoint."""

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    thread_id: str = Field(..., description="LangGraph thread id (session id)")
    parent_checkpoint_id: str | None = Field(None, description="Parent checkpoint id, if any")
    created_at: str | None = Field(None, description="Checkpoint creation timestamp")
    source: str | None = Field(None, description="Checkpoint source (input, loop, update, fork)")
    step: int | None = Field(None, description="Graph step number when checkpoint was created")


class CheckpointDetail(CheckpointSummary):
    """Detailed checkpoint metadata including channel information."""

    channel_versions: dict[str, Any] = Field(default_factory=dict, description="Channel version map")
    updated_channels: list[str] | None = Field(None, description="Channels updated in this checkpoint")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Raw checkpoint metadata")


class StateHistoryEntry(BaseModel):
    """A graph state snapshot from checkpoint history."""

    checkpoint_id: str | None = Field(None, description="Checkpoint id for this snapshot")
    thread_id: str | None = Field(None, description="LangGraph thread id (session id)")
    created_at: str | None = Field(None, description="Snapshot creation timestamp")
    next: list[str] = Field(default_factory=list, description="Next nodes to execute")
    source: str | None = Field(None, description="Checkpoint source")
    step: int | None = Field(None, description="Graph step number")
    values: dict[str, Any] = Field(default_factory=dict, description="State values at this snapshot")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Snapshot metadata")


def extract_checkpoint_id(config: dict[str, Any] | None) -> str | None:
    """Return checkpoint_id from a RunnableConfig configurable section."""
    if not config:
        return None
    configurable = config.get("configurable") or {}
    checkpoint_id = configurable.get("checkpoint_id")
    return str(checkpoint_id) if checkpoint_id is not None else None


def extract_thread_id(config: dict[str, Any] | None) -> str | None:
    """Return thread_id from a RunnableConfig configurable section."""
    if not config:
        return None
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id is not None else None


def _get_tuple_field(checkpoint_tuple: CheckpointTuple | dict[str, Any], field: str) -> Any:
    """Read a field from a LangGraph CheckpointTuple or test double dict."""
    if isinstance(checkpoint_tuple, dict):
        return checkpoint_tuple.get(field)
    return getattr(checkpoint_tuple, field, None)


def serialize_checkpoint_summary(checkpoint_tuple: CheckpointTuple | dict[str, Any]) -> CheckpointSummary:
    """Convert a LangGraph CheckpointTuple into a CheckpointSummary."""
    metadata = dict(_get_tuple_field(checkpoint_tuple, "metadata") or {})
    checkpoint = _get_tuple_field(checkpoint_tuple, "checkpoint") or {}
    return CheckpointSummary(
        checkpoint_id=checkpoint.get("id", ""),
        thread_id=extract_thread_id(_get_tuple_field(checkpoint_tuple, "config")) or "",
        parent_checkpoint_id=extract_checkpoint_id(_get_tuple_field(checkpoint_tuple, "parent_config")),
        created_at=checkpoint.get("ts"),
        source=metadata.get("source"),
        step=metadata.get("step"),
    )


def serialize_checkpoint_detail(checkpoint_tuple: CheckpointTuple | dict[str, Any]) -> CheckpointDetail:
    """Convert a LangGraph CheckpointTuple into a CheckpointDetail."""
    summary = serialize_checkpoint_summary(checkpoint_tuple)
    checkpoint = _get_tuple_field(checkpoint_tuple, "checkpoint") or {}
    metadata = dict(_get_tuple_field(checkpoint_tuple, "metadata") or {})
    return CheckpointDetail(
        **summary.model_dump(),
        channel_versions=dict(checkpoint.get("channel_versions") or {}),
        updated_channels=checkpoint.get("updated_channels"),
        metadata=metadata,
    )


def _serialize_state_values(values: dict[str, Any] | Any) -> dict[str, Any]:
    """Make graph state values JSON-serializable for API responses."""
    if not isinstance(values, dict):
        return {"value": str(values)}

    serialized: dict[str, Any] = {}
    for key, value in values.items():
        if key == "messages" and value:
            openai_messages = convert_to_openai_messages(value)
            serialized[key] = [
                {"role": message.get("role"), "content": message.get("content")}
                for message in openai_messages
                if message.get("content")
            ]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, (list, dict)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def serialize_state_history_entry(snapshot: StateSnapshot) -> StateHistoryEntry:
    """Convert a LangGraph StateSnapshot into a StateHistoryEntry."""
    metadata = dict(snapshot.metadata or {})
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    return StateHistoryEntry(
        checkpoint_id=extract_checkpoint_id(snapshot.config),
        thread_id=extract_thread_id(snapshot.config),
        created_at=snapshot.created_at,
        next=list(snapshot.next or ()),
        source=metadata.get("source"),
        step=metadata.get("step"),
        values=_serialize_state_values(values),
        metadata=metadata,
    )
