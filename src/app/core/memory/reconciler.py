"""Reconcile new facts with existing memories using LLM-guided actions."""

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.models import MemoryReconcileOutput
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


class MemoryReconciler:
    """Compare extracted facts with existing memories and choose ADD/UPDATE/DELETE/NONE."""

    def __init__(self, app_settings: Settings, chat_model: BaseChatModel) -> None:
        self._settings = app_settings
        self._chat_model = chat_model.with_structured_output(MemoryReconcileOutput)

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

        prompt = build_memory_reconcile_prompt(existing_memories, new_facts)
        try:
            response = await self._chat_model.ainvoke(
                [
                    SystemMessage(content="You reconcile user long-term memory updates."),
                    HumanMessage(content=prompt),
                ]
            )
            if not isinstance(response, MemoryReconcileOutput):
                return []

            return self._parse_actions(response, id_mapping)
        except Exception:
            logger.exception("memory_reconciliation_failed")
            return []

    def _parse_actions(self, payload: MemoryReconcileOutput, id_mapping: dict[str, str]) -> list[MemoryAction]:
        actions: list[MemoryAction] = []
        for item in payload.memory:
            event = item.event.upper()
            text = item.text.strip()
            if event not in VALID_EVENTS or not text:
                continue

            temp_id = item.id.strip()
            previous_memory = item.old_memory
            previous_memory_text = previous_memory.strip() if previous_memory else None

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
