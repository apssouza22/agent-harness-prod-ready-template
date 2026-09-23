"""Unit tests for MemoryService."""

from unittest.mock import AsyncMock, patch

import pytest

from src.app.core.memory.memory import MemoryService


@pytest.fixture
def memory_service() -> MemoryService:
    return MemoryService()


@pytest.mark.asyncio
async def test_search_returns_formatted_memories(memory_service: MemoryService) -> None:
    mock_engine = AsyncMock()
    mock_engine.search.return_value = {
        "results": [{"memory": "User prefers dark mode"}, {"memory": "User works in Python"}]
    }
    memory_service._engine = mock_engine

    result = await memory_service.search(user_id=1, query="preferences")

    assert result == "* User prefers dark mode\n* User works in Python"
    mock_engine.search.assert_awaited_once_with(user_id="1", query="preferences")


@pytest.mark.asyncio
async def test_search_returns_empty_string_on_error(memory_service: MemoryService) -> None:
    mock_engine = AsyncMock()
    mock_engine.search.side_effect = RuntimeError("connection failed")
    memory_service._engine = mock_engine

    result = await memory_service.search(user_id=1, query="preferences")

    assert result == ""


@pytest.mark.asyncio
async def test_add_delegates_to_engine(memory_service: MemoryService) -> None:
    mock_engine = AsyncMock()
    memory_service._engine = mock_engine
    messages = [{"role": "user", "content": "hello"}]
    metadata = {"session_id": "abc"}

    await memory_service.add(user_id=42, messages=messages, metadata=metadata)

    mock_engine.add.assert_awaited_once_with(messages, user_id="42", metadata=metadata)


def test_schedule_add_creates_background_task(memory_service: MemoryService) -> None:
    with patch("src.app.core.memory.memory.asyncio.create_task") as mock_create_task:
        memory_service.schedule_add(user_id=1, messages=[{"role": "user", "content": "hi"}])

    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_search_returns_empty_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "false")
    from src.app.core.common import config as config_module

    disabled_settings = config_module.Settings()
    service = MemoryService(disabled_settings)

    result = await service.search(user_id=1, query="preferences")

    assert result == ""
