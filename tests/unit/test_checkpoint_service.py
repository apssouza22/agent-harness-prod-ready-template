"""Unit tests for CheckpointService."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.checkpoint.models import CheckpointDetail, CheckpointSummary, StateHistoryEntry
from src.app.core.checkpoint.service import CheckpointService
from src.app.core.graph.compiled import StateGraphCompiled


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.ENVIRONMENT.value = "test"
    return settings


@pytest.fixture
def checkpoint_service(mock_settings) -> CheckpointService:
    return CheckpointService(mock_settings, connection_pool=AsyncMock())


@pytest.fixture
def mock_checkpointer(checkpoint_service: CheckpointService) -> AsyncMock:
    checkpointer = AsyncMock()
    checkpoint_service._checkpointer = checkpointer
    return checkpointer


def _make_checkpoint_tuple(
    *,
    checkpoint_id: str = "cp-1",
    thread_id: str = "session-1",
    parent_id: str | None = None,
    ts: str = "2026-01-01T00:00:00Z",
):
    return {
        "config": {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}},
        "checkpoint": {
            "id": checkpoint_id,
            "ts": ts,
            "channel_versions": {"messages": 1},
            "updated_channels": ["messages"],
        },
        "metadata": {"source": "loop", "step": 2},
        "parent_config": {"configurable": {"checkpoint_id": parent_id}} if parent_id else None,
        "pending_writes": None,
    }


def _make_state_snapshot(
    *,
    checkpoint_id: str = "cp-1",
    thread_id: str = "session-1",
    created_at: str = "2026-01-01T00:00:00Z",
):
    return MagicMock(
        config={"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}},
        created_at=created_at,
        next=("chat",),
        metadata={"source": "loop", "step": 2},
        values={"messages": [], "long_term_memory": ""},
    )


@pytest.mark.asyncio
async def test_list_checkpoints_returns_summaries(
    checkpoint_service: CheckpointService,
    mock_checkpointer: AsyncMock,
) -> None:
    tuples = [_make_checkpoint_tuple(checkpoint_id="cp-2"), _make_checkpoint_tuple(checkpoint_id="cp-1")]

    async def _alist(*_args, **_kwargs):
        for item in tuples:
            yield item

    mock_checkpointer.alist = _alist

    result = await checkpoint_service.list_checkpoints("session-1", limit=10)

    assert len(result) == 2
    assert all(isinstance(item, CheckpointSummary) for item in result)
    assert result[0].checkpoint_id == "cp-2"
    assert result[0].thread_id == "session-1"
    assert result[0].step == 2


@pytest.mark.asyncio
async def test_get_checkpoint_returns_detail(
    checkpoint_service: CheckpointService,
    mock_checkpointer: AsyncMock,
) -> None:
    mock_checkpointer.aget_tuple.return_value = _make_checkpoint_tuple(parent_id="cp-0")

    result = await checkpoint_service.get_checkpoint("session-1", "cp-1")

    assert isinstance(result, CheckpointDetail)
    assert result.checkpoint_id == "cp-1"
    assert result.parent_checkpoint_id == "cp-0"
    assert result.updated_channels == ["messages"]
    mock_checkpointer.aget_tuple.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_checkpoint_returns_none_when_missing(
    checkpoint_service: CheckpointService,
    mock_checkpointer: AsyncMock,
) -> None:
    mock_checkpointer.aget_tuple.return_value = None

    result = await checkpoint_service.get_checkpoint("session-1", "missing")

    assert result is None


@pytest.mark.asyncio
async def test_clear_session_uses_adelete_thread(
    checkpoint_service: CheckpointService,
    mock_checkpointer: AsyncMock,
) -> None:
    await checkpoint_service.clear_session("session-1")

    mock_checkpointer.adelete_thread.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_copy_session_delegates_to_checkpointer(
    checkpoint_service: CheckpointService,
    mock_checkpointer: AsyncMock,
) -> None:
    await checkpoint_service.copy_session("source", "target")

    mock_checkpointer.acopy_thread.assert_awaited_once_with("source", "target")


@pytest.mark.asyncio
async def test_prune_sessions_delegates_to_checkpointer(
    checkpoint_service: CheckpointService,
    mock_checkpointer: AsyncMock,
) -> None:
    await checkpoint_service.prune_sessions(["s-1", "s-2"], strategy="keep_latest")

    mock_checkpointer.aprune.assert_awaited_once_with(["s-1", "s-2"], strategy="keep_latest")


@pytest.mark.asyncio
async def test_get_state_history_serializes_snapshots(
    checkpoint_service: CheckpointService,
) -> None:
    mock_graph = AsyncMock(spec=StateGraphCompiled)
    mock_graph.aget_state_history.return_value = [_make_state_snapshot()]

    result = await checkpoint_service.get_state_history("session-1", mock_graph, limit=5)

    assert len(result) == 1
    assert isinstance(result[0], StateHistoryEntry)
    assert result[0].checkpoint_id == "cp-1"
    assert result[0].next == ["chat"]
    mock_graph.aget_state_history.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_checkpointer_raises_when_unavailable(mock_settings) -> None:
    from src.app.core.common.config import Environment

    mock_settings.ENVIRONMENT = Environment.PRODUCTION
    service = CheckpointService(mock_settings, connection_pool=None)
    service._checkpointer = None

    with pytest.raises(RuntimeError, match="checkpointer unavailable"):
        await service.list_checkpoints("session-1")
