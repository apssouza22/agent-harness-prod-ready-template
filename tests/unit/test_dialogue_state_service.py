"""Unit tests for DialogueStateService."""

from unittest.mock import AsyncMock, patch

import pytest

from src.app.core.dialogue_state.models import DialogueState
from src.app.core.dialogue_state.service import DialogueStateService


@pytest.fixture
def dialogue_state_service() -> DialogueStateService:
    return DialogueStateService()


@pytest.mark.asyncio
async def test_get_formatted_delegates_to_engine(dialogue_state_service: DialogueStateService) -> None:
    mock_engine = AsyncMock()
    mock_engine.get_formatted.return_value = "- Topic: travel planning"
    dialogue_state_service._engine = mock_engine

    result = await dialogue_state_service.get_formatted(session_id="session-1")

    assert result == "- Topic: travel planning"
    mock_engine.get_formatted.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_get_formatted_returns_empty_string_on_error(
    dialogue_state_service: DialogueStateService,
) -> None:
    mock_engine = AsyncMock()
    mock_engine.get_formatted.side_effect = RuntimeError("connection failed")
    dialogue_state_service._engine = mock_engine

    result = await dialogue_state_service.get_formatted(session_id="session-1")

    assert result == ""


@pytest.mark.asyncio
async def test_update_delegates_to_engine(dialogue_state_service: DialogueStateService) -> None:
    mock_engine = AsyncMock()
    dialogue_state_service._engine = mock_engine
    messages = [{"role": "user", "content": "hello"}]

    await dialogue_state_service.update(session_id="session-1", user_id=42, messages=messages)

    mock_engine.update.assert_awaited_once_with("session-1", "42", messages)


def test_schedule_update_creates_background_task(dialogue_state_service: DialogueStateService) -> None:
    with patch("src.app.core.dialogue_state.service.asyncio.create_task") as mock_create_task:
        dialogue_state_service.schedule_update(
            session_id="session-1",
            user_id=1,
            messages=[{"role": "user", "content": "hi"}],
        )

    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_get_formatted_returns_empty_when_disabled(
    dialogue_state_service: DialogueStateService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dialogue_state_service._settings, "DIALOGUE_STATE_ENABLED", False)

    result = await dialogue_state_service.get_formatted(session_id="session-1")

    assert result == ""


def test_dialogue_state_to_prompt_text_formats_fields() -> None:
    state = DialogueState(
        topic="travel planning",
        active_goals=["book a hotel"],
        slots={"destination": "Lisbon"},
        pending_clarifications=["What dates?"],
        entities_in_focus=["Lisbon"],
        conversation_phase="information_gathering",
        summary="User wants help booking a hotel in Lisbon.",
    )

    prompt_text = state.to_prompt_text()

    assert "Topic: travel planning" in prompt_text
    assert "destination=Lisbon" in prompt_text
    assert "Pending clarifications: What dates?" in prompt_text
