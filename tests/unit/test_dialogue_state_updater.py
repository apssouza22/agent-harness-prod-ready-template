"""Unit tests for DialogueStateUpdater."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.dialogue_state.models import DialogueState
from src.app.core.dialogue_state.updater import DialogueStateUpdater


@pytest.fixture
def updater() -> DialogueStateUpdater:
    from src.app.core.common.config import settings

    return DialogueStateUpdater(settings)


@pytest.mark.asyncio
async def test_update_state_returns_previous_state_for_empty_conversation(
    updater: DialogueStateUpdater,
) -> None:
    previous_state = DialogueState(topic="existing topic")

    result = await updater.update_state(previous_state, [])

    assert result == previous_state


@pytest.mark.asyncio
async def test_update_state_parses_llm_json(updater: DialogueStateUpdater) -> None:
    previous_state = DialogueState()
    messages = [{"role": "user", "content": "I need a hotel in Lisbon"}]
    updated_payload = {
        "topic": "hotel booking",
        "active_goals": ["find a hotel in Lisbon"],
        "slots": {"destination": "Lisbon"},
        "pending_clarifications": ["What dates?"],
        "entities_in_focus": ["Lisbon"],
        "conversation_phase": "information_gathering",
        "summary": "User wants a hotel in Lisbon.",
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps(updated_payload)
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_response

    captured_kwargs: dict[str, object] = {}

    def capture_make_chat_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_llm

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "src.app.core.dialogue_state.updater.make_chat_model",
            capture_make_chat_model,
        )
        result = await updater.update_state(previous_state, messages)

    assert captured_kwargs.get("bifrost_agent") == "agent_1"

    assert result.topic == "hotel booking"
    assert result.slots["destination"] == "Lisbon"
    assert result.pending_clarifications == ["What dates?"]
