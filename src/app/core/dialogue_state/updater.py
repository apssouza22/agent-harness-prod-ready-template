"""LLM-based dialogue state updater."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.dialogue_state.models import DialogueState, DialogueStateLLMOutput
from src.app.core.dialogue_state.prompts import (
    DEFAULT_DIALOGUE_STATE_UPDATE_PROMPT,
    build_dialogue_state_update_prompt,
)


def _parse_messages_for_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            lines.append(f"{str(role).capitalize()}: {content}")
    return "\n".join(lines)


class DialogueStateUpdater:
    """Merge conversation turns into structured dialogue state."""

    def __init__(self, app_settings: Settings, chat_model: BaseChatModel) -> None:
        self._settings = app_settings
        self._chat_model = chat_model

    async def update_state(
        self,
        previous_state: DialogueState,
        messages: list[dict[str, Any]],
    ) -> DialogueState:
        """Return updated dialogue state from the latest conversation turn."""
        conversation = _parse_messages_for_prompt(messages)
        if not conversation.strip():
            return previous_state

        previous_state_json = json.dumps(previous_state.model_dump(), ensure_ascii=True)
        user_prompt = build_dialogue_state_update_prompt(previous_state_json, conversation)

        try:
            updated_output = await self._chat_model.ainvoke(
                [
                    SystemMessage(content=DEFAULT_DIALOGUE_STATE_UPDATE_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            if not isinstance(updated_output, DialogueStateLLMOutput):
                return previous_state
            return updated_output.to_dialogue_state()
        except Exception:
            logger.exception("dialogue_state_update_failed")
            return previous_state
