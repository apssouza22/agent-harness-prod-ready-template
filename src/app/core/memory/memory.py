"""Long-term memory management using local pgvector storage.

This module provides a MemoryService class for managing long-term memory operations
including initialization, search, and updates using pgvector with LLM fact extraction.
"""

import asyncio
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.engine import LongTermMemoryEngine


class MemoryService:
    """Service for long-term memory operations using pgvector and LLM extraction.

    Encapsulates the LongTermMemoryEngine and exposes typed methods for
    searching and updating user memories.
    """

    def __init__(
        self,
        app_settings: Settings,
        chat_model: BaseChatModel,
    ) -> None:
        self._settings = app_settings
        self._chat_model = chat_model
        self._engine: Optional[LongTermMemoryEngine] = None

    async def _get_engine(self) -> LongTermMemoryEngine:
        """Lazily initialize and return the long-term memory engine."""
        if self._engine is None:
            self._engine = LongTermMemoryEngine(self._settings, chat_model=self._chat_model)
            await self._engine.initialize()
            logger.info(
                "long_term_memory_initialized",
                collection_name=self._settings.LONG_TERM_MEMORY_COLLECTION_NAME,
            )
        return self._engine

    async def search(self, user_id: int, query: str) -> str:
        """Get relevant memories for a user and query.

        Args:
            user_id: The user ID to search memories for.
            query: The query to search for relevant memories.

        Returns:
            Formatted string of relevant memories, or empty string on error.
        """
        if not self._settings.LONG_TERM_MEMORY_ENABLED:
            return ""

        try:
            engine = await self._get_engine()
            results = await engine.search(user_id=str(user_id), query=query)
            memory_result = "\n".join(f"* {result['memory']}" for result in results["results"])
            logger.debug("relevant_memory_retrieved", user_id=user_id, result_count=len(results["results"]))
            return memory_result
        except Exception as e:
            logger.exception("failed_to_get_relevant_memory", user_id=user_id, query=query, error=str(e))
            return ""

    async def add(self, user_id: int, messages: list[dict], metadata: Optional[dict] = None) -> None:
        """Update long-term memory with new messages.

        Args:
            user_id: The user ID to update memory for.
            messages: The messages to add to memory.
            metadata: Optional metadata to include with the memory update.
        """
        if not self._settings.LONG_TERM_MEMORY_ENABLED:
            return

        try:
            engine = await self._get_engine()
            await engine.add(messages, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully", user_id=user_id)
        except Exception as e:
            logger.exception("failed_to_update_long_term_memory", user_id=user_id, error=str(e))

    def schedule_add(self, user_id: int, messages: list[dict], metadata: Optional[dict] = None) -> None:
        """Schedule a memory update in the background without blocking the response.

        Args:
            user_id: The user ID to update memory for.
            messages: The messages to add to memory.
            metadata: Optional metadata to include with the memory update.
        """
        if not self._settings.LONG_TERM_MEMORY_ENABLED:
            return

        asyncio.create_task(self.add(user_id, messages, metadata))
