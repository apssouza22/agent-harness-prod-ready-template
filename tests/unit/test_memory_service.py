"""Unit tests for MemoryService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.core.memory.memory import MemoryService


@pytest.fixture
def memory_service() -> MemoryService:
    return MemoryService()


@pytest.mark.asyncio
async def test_search_returns_formatted_memories(memory_service: MemoryService) -> None:
    mock_memory = AsyncMock()
    mock_memory.search.return_value = {
        "results": [{"memory": "User prefers dark mode"}, {"memory": "User works in Python"}]
    }
    memory_service._memory = mock_memory

    result = await memory_service.search(user_id=1, query="preferences")

    assert result == "* User prefers dark mode\n* User works in Python"
    mock_memory.search.assert_awaited_once_with(user_id="1", query="preferences")


@pytest.mark.asyncio
async def test_search_returns_empty_string_on_error(memory_service: MemoryService) -> None:
    mock_memory = AsyncMock()
    mock_memory.search.side_effect = RuntimeError("connection failed")
    memory_service._memory = mock_memory

    result = await memory_service.search(user_id=1, query="preferences")

    assert result == ""


@pytest.mark.asyncio
async def test_add_delegates_to_mem0(memory_service: MemoryService) -> None:
    mock_memory = AsyncMock()
    memory_service._memory = mock_memory
    messages = [{"role": "user", "content": "hello"}]
    metadata = {"session_id": "abc"}

    await memory_service.add(user_id=42, messages=messages, metadata=metadata)

    mock_memory.add.assert_awaited_once_with(messages, user_id="42", metadata=metadata)


def test_schedule_add_creates_background_task(memory_service: MemoryService) -> None:
    with patch("src.app.core.memory.memory.asyncio.create_task") as mock_create_task:
        memory_service.schedule_add(user_id=1, messages=[{"role": "user", "content": "hi"}])

    mock_create_task.assert_called_once()


def test_build_config_includes_custom_instructions_when_set(memory_service: MemoryService) -> None:
    custom_instructions = "Extract user preferences and goals. Exclude personal identifiers."
    mock_settings = MagicMock()
    mock_settings.LONG_TERM_MEMORY_COLLECTION_NAME = "longterm_memory"
    mock_settings.POSTGRES_DB = "mydb"
    mock_settings.POSTGRES_USER = "user"
    mock_settings.POSTGRES_PASSWORD = "pass"
    mock_settings.POSTGRES_HOST = "localhost"
    mock_settings.POSTGRES_PORT = 5432
    mock_settings.LONG_TERM_MEMORY_MODEL = "gpt-5-nano"
    mock_settings.LONG_TERM_MEMORY_EMBEDDER_MODEL = "text-embedding-3-small"
    mock_settings.LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS = custom_instructions
    memory_service._settings = mock_settings

    with patch("src.app.core.memory.memory.build_mem0_openai_config", return_value={"api_key": "test-key"}):
        config = memory_service._build_config()

    assert config["custom_instructions"] == custom_instructions


def test_build_config_omits_custom_instructions_when_unset(memory_service: MemoryService) -> None:
    mock_settings = MagicMock()
    mock_settings.LONG_TERM_MEMORY_COLLECTION_NAME = "longterm_memory"
    mock_settings.POSTGRES_DB = "mydb"
    mock_settings.POSTGRES_USER = "user"
    mock_settings.POSTGRES_PASSWORD = "pass"
    mock_settings.POSTGRES_HOST = "localhost"
    mock_settings.POSTGRES_PORT = 5432
    mock_settings.LONG_TERM_MEMORY_MODEL = "gpt-5-nano"
    mock_settings.LONG_TERM_MEMORY_EMBEDDER_MODEL = "text-embedding-3-small"
    mock_settings.LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS = None
    memory_service._settings = mock_settings

    with patch("src.app.core.memory.memory.build_mem0_openai_config", return_value={"api_key": "test-key"}):
        config = memory_service._build_config()

    assert "custom_instructions" not in config
