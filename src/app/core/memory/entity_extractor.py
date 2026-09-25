"""Extract entities from memory text for entity-linked retrieval boosting."""

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.entities import MemoryEntity, normalize_entity_name
from src.app.core.memory.prompts import DEFAULT_ENTITY_EXTRACTION_PROMPT


def _strip_code_blocks(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class EntityExtractor:
    """Extract people, organizations, projects, tools, and places from memory text."""

    def __init__(self, app_settings: Settings, chat_model: BaseChatModel) -> None:
        self._settings = app_settings
        self._chat_model = chat_model

    async def extract_entities(self, text: str) -> list[MemoryEntity]:
        """Return structured entities referenced in the memory text."""
        cleaned = text.strip()
        if not cleaned:
            return []

        if not self._settings.LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED:
            return []

        try:
            response = await self._chat_model.ainvoke(
                [
                    SystemMessage(content=DEFAULT_ENTITY_EXTRACTION_PROMPT),
                    HumanMessage(content=f"Text:\n{cleaned}\n\nOutput:"),
                ]
            )
            raw_content = _strip_code_blocks(str(response.content))
            if not raw_content:
                return []

            parsed = json.loads(raw_content)
            return self._parse_entities(parsed)
        except Exception:
            logger.exception("memory_entity_extraction_failed")
            return []

    def _parse_entities(self, payload: dict[str, Any]) -> list[MemoryEntity]:
        entities: list[MemoryEntity] = []
        seen: set[str] = set()

        for item in payload.get("entities", []):
            if isinstance(item, str):
                name = item.strip()
                entity_type = "unknown"
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                entity_type = str(item.get("type", "unknown")).strip() or "unknown"
            else:
                continue

            if not name:
                continue

            normalized = normalize_entity_name(name)
            if normalized in seen:
                continue

            seen.add(normalized)
            entities.append(MemoryEntity(name=name, entity_type=entity_type))

        return entities
