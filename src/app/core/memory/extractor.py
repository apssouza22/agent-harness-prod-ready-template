"""LLM-based fact extraction for long-term memory ingestion."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.models import FactExtractionOutput
from src.app.core.memory.prompts import build_fact_extraction_prompt


def _parse_messages_for_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            lines.append(f"{str(role).capitalize()}: {content}")
    return "\n".join(lines)


class FactExtractor:
    """Extract durable user facts from conversation messages."""

    def __init__(self, app_settings: Settings, chat_model: BaseChatModel) -> None:
        self._settings = app_settings
        self._chat_model = chat_model.with_structured_output(FactExtractionOutput)

    async def extract_facts(self, messages: list[dict[str, Any]]) -> list[str]:
        """Return extracted fact strings from a conversation transcript."""
        conversation = _parse_messages_for_prompt(messages)
        if not conversation.strip():
            return []

        system_prompt = build_fact_extraction_prompt(self._settings.LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS)
        user_prompt = f"Input:\n{conversation}\n\nOutput:"

        try:
            response = await self._chat_model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            if not isinstance(response, FactExtractionOutput):
                return []

            return [fact.strip() for fact in response.facts if fact.strip()]
        except Exception:
            logger.exception("memory_fact_extraction_failed")
            return []
