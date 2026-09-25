"""Unit tests for DialogueStateUpdater."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.dialogue_state.models import DialogueSlot, DialogueState, DialogueStateLLMOutput
from src.app.core.dialogue_state.updater import DialogueStateUpdater


@pytest.fixture
def updater() -> DialogueStateUpdater:
    from src.app.core.common.config import settings

    return DialogueStateUpdater(settings, chat_model=MagicMock())


@pytest.mark.asyncio
async def test_update_state_returns_previous_state_for_empty_conversation(
    updater: DialogueStateUpdater,
) -> None:
    previous_state = DialogueState(topic="existing topic")

    result = await updater.update_state(previous_state, [])

    assert result == previous_state


@pytest.mark.asyncio
async def test_update_state_parses_llm_json() -> None:
    from src.app.core.common.config import settings

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

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = DialogueStateLLMOutput(
        topic=updated_payload["topic"],
        active_goals=updated_payload["active_goals"],
        slots=[DialogueSlot(key="destination", value="Lisbon")],
        pending_clarifications=updated_payload["pending_clarifications"],
        entities_in_focus=updated_payload["entities_in_focus"],
        conversation_phase=updated_payload["conversation_phase"],
        summary=updated_payload["summary"],
    )
    updater = DialogueStateUpdater(settings, chat_model=mock_chat_model)

    result = await updater.update_state(previous_state, messages)

    mock_chat_model.ainvoke.assert_awaited_once()
    assert result.topic == "hotel booking"
    assert result.slots["destination"] == "Lisbon"
    assert result.pending_clarifications == ["What dates?"]


def test_dialogue_state_llm_output_converts_slots_to_dict() -> None:
    output = DialogueStateLLMOutput(
        topic="travel",
        slots=[
            DialogueSlot(key="destination", value="Lisbon"),
            DialogueSlot(key="", value="ignored"),
        ],
    )

    state = output.to_dialogue_state()

    assert state.topic == "travel"
    assert state.slots == {"destination": "Lisbon"}
