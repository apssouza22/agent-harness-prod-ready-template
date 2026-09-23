"""LLM-based dialogue state updater."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.dialogue_state.config_builder import resolve_dialogue_state_provider
from src.app.core.dialogue_state.models import DialogueState
from src.app.core.dialogue_state.prompts import (
    DEFAULT_DIALOGUE_STATE_UPDATE_PROMPT,
    build_dialogue_state_update_prompt,
)
from src.app.core.llm.factory import make_chat_model, resolve_model_identifier


def _strip_code_blocks(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


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

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings

    async def update_state(
        self,
        previous_state: DialogueState,
        messages: list[dict[str, Any]],
    ) -> DialogueState:
        """Return updated dialogue state from the latest conversation turn."""
        conversation = _parse_messages_for_prompt(messages)
        if not conversation.strip():
            return previous_state

        provider = resolve_dialogue_state_provider(
            self._settings.DIALOGUE_STATE_LLM_PROVIDER,
            self._settings.DIALOGUE_STATE_MODEL,
            fallback_provider=self._settings.DEFAULT_LLM_PROVIDER,
        )
        model_name = resolve_model_identifier(
            self._settings.DIALOGUE_STATE_MODEL,
            provider,
            self._settings,
        )
        llm = make_chat_model(
            model_name,
            app_settings=self._settings,
            max_tokens=self._settings.MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        previous_state_json = json.dumps(previous_state.model_dump(), ensure_ascii=True)
        user_prompt = build_dialogue_state_update_prompt(previous_state_json, conversation)

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content=DEFAULT_DIALOGUE_STATE_UPDATE_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw_content = _strip_code_blocks(str(response.content))
            if not raw_content:
                return previous_state

            parsed = json.loads(raw_content)
            return DialogueState.model_validate(parsed)
        except Exception:
            logger.exception("dialogue_state_update_failed")
            return previous_state
