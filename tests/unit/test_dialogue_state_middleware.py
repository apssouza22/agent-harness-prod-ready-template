"""Unit tests for DialogueStateMiddleware."""

from unittest.mock import AsyncMock

import pytest

from src.app.core.dialogue_state.middleware import DialogueStateMiddleware
from src.app.core.common.model.message import Message
from src.app.core.middleware.types import AgentContext


@pytest.fixture
def middleware() -> DialogueStateMiddleware:
    service = AsyncMock()
    service.get_formatted.return_value = "- Topic: onboarding"
    return DialogueStateMiddleware(dialogue_state=service)


@pytest.mark.asyncio
async def test_before_invoke_sets_dialogue_state_metadata(middleware: DialogueStateMiddleware) -> None:
    ctx = AgentContext(
        messages=[Message(role="user", content="hello")],
        session_id="session-1",
        user_id=1,
        config={},
        agent_name="Chatbot",
    )

    result = await middleware.before_invoke(ctx)

    assert result is None
    assert ctx.metadata["dialogue_state"] == "- Topic: onboarding"
    middleware._dialogue_state.get_formatted.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
async def test_after_invoke_schedules_update(middleware: DialogueStateMiddleware) -> None:
    ctx = AgentContext(
        messages=[Message(role="user", content="hello")],
        session_id="session-1",
        user_id=7,
        config={},
        agent_name="Chatbot",
    )
    invoke_result = [Message(role="assistant", content="hi there")]

    result = await middleware.after_invoke(ctx, invoke_result)

    assert result == invoke_result
    middleware._dialogue_state.schedule_update.assert_called_once_with(
        "session-1",
        7,
        [{"role": "assistant", "content": "hi there"}],
    )
