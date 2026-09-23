"""Reconcile new facts with existing memories using LLM-guided actions."""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.llm.factory import make_chat_model, resolve_model_identifier
from src.app.core.memory.config_builder import resolve_memory_provider
from src.app.core.memory.prompts import build_memory_reconcile_prompt

MemoryEvent = Literal["ADD", "UPDATE", "DELETE", "NONE"]
VALID_EVENTS = frozenset({"ADD", "UPDATE", "DELETE", "NONE"})


@dataclass(frozen=True)
class MemoryAction:
    """A memory mutation decided by the reconciliation step."""

    event: MemoryEvent
    text: str
    memory_id: str | None = None
    previous_memory: str | None = None


def _strip_code_blocks(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class MemoryReconciler:
    """Compare extracted facts with existing memories and choose ADD/UPDATE/DELETE/NONE."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings

    async def reconcile(
        self,
        existing_memories: list[dict[str, str]],
        new_facts: list[str],
        *,
        id_mapping: dict[str, str],
    ) -> list[MemoryAction]:
        """Return validated memory actions for the extracted facts."""
        if not new_facts:
            return []

        if not existing_memories:
            return [MemoryAction(event="ADD", text=fact) for fact in new_facts]

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

        prompt = build_memory_reconcile_prompt(existing_memories, new_facts)
        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You reconcile user long-term memory updates."),
                    HumanMessage(content=prompt),
                ]
            )
            raw_content = _strip_code_blocks(str(response.content))
            if not raw_content:
                return []

            parsed = json.loads(raw_content)
            return self._parse_actions(parsed, id_mapping)
        except Exception:
            logger.exception("memory_reconciliation_failed")
            return []

    def _parse_actions(self, payload: dict[str, Any], id_mapping: dict[str, str]) -> list[MemoryAction]:
        actions: list[MemoryAction] = []
        for item in payload.get("memory", []):
            if not isinstance(item, dict):
                continue

            event = str(item.get("event", "")).upper()
            text = str(item.get("text", "")).strip()
            if event not in VALID_EVENTS or not text:
                continue

            temp_id = str(item.get("id", "")).strip()
            previous_memory = item.get("old_memory")
            previous_memory_text = str(previous_memory).strip() if previous_memory else None

            if event == "ADD":
                actions.append(MemoryAction(event="ADD", text=text))
                continue

            real_id = id_mapping.get(temp_id)
            if not real_id:
                logger.warning("memory_reconcile_unknown_id", temp_id=temp_id, memory_event=event)
                continue

            actions.append(
                MemoryAction(
                    event=event,  # type: ignore[arg-type]
                    text=text,
                    memory_id=real_id,
                    previous_memory=previous_memory_text,
                )
            )
        return actions
