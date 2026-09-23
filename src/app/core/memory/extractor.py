"""LLM-based fact extraction for long-term memory ingestion."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.llm.factory import make_chat_model, resolve_model_identifier
from src.app.core.memory.config_builder import resolve_memory_provider
from src.app.core.memory.prompts import build_fact_extraction_prompt


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


class FactExtractor:
    """Extract durable user facts from conversation messages."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings

    async def extract_facts(self, messages: list[dict[str, Any]]) -> list[str]:
        """Return extracted fact strings from a conversation transcript."""
        conversation = _parse_messages_for_prompt(messages)
        if not conversation.strip():
            return []

        provider = resolve_memory_provider(
            self._settings.LONG_TERM_MEMORY_LLM_PROVIDER,
            self._settings.LONG_TERM_MEMORY_MODEL,
            fallback_provider=self._settings.DEFAULT_LLM_PROVIDER,
        )
        model_name = resolve_model_identifier(
            self._settings.LONG_TERM_MEMORY_MODEL,
            provider,
            self._settings,
        )
        llm = make_chat_model(
            model_name,
            app_settings=self._settings,
            max_tokens=self._settings.MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        system_prompt = build_fact_extraction_prompt(self._settings.LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS)
        user_prompt = f"Input:\n{conversation}\n\nOutput:"

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw_content = _strip_code_blocks(str(response.content))
            if not raw_content:
                return []

            parsed = json.loads(raw_content)
            facts = parsed.get("facts", [])
            if not isinstance(facts, list):
                return []

            return [str(fact).strip() for fact in facts if str(fact).strip()]
        except Exception:
            logger.exception("memory_fact_extraction_failed")
            return []
