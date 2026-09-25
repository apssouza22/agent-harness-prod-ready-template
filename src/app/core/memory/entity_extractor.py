"""Extract entities from memory text for entity-linked retrieval boosting."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.entities import MemoryEntity
from src.app.core.memory.models import EntityExtractionOutput
from src.app.core.memory.prompts import DEFAULT_ENTITY_EXTRACTION_PROMPT


class EntityExtractor:
    """Extract people, organizations, projects, tools, and places from memory text."""

    def __init__(self, app_settings: Settings, chat_model: BaseChatModel) -> None:
        self._settings = app_settings
        self._chat_model = chat_model.with_structured_output(EntityExtractionOutput)

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
            if not isinstance(response, EntityExtractionOutput):
                return []

            return response.to_memory_entities()
        except Exception:
            logger.exception("memory_entity_extraction_failed")
            return []
