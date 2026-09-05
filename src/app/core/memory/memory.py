"""Long-term memory management using mem0 and pgvector.

This module provides a MemoryService class for managing long-term memory operations
including initialization, search, and updates using the mem0 library with PostgreSQL/pgvector backend.
"""

import asyncio
from typing import Any, Optional

from mem0 import AsyncMemory

from src.app.core.common.config import Settings, settings
from src.app.core.common.logging import logger
from src.app.core.llm.factory import build_mem0_openai_config


class MemoryService:
    """Service for long-term memory operations using mem0 and pgvector.

    Encapsulates the AsyncMemory singleton and exposes typed methods for
    searching and updating user memories.
    """

    def __init__(self, app_settings: Settings | None = None) -> None:
        self._settings = app_settings or settings
        self._memory: Optional[AsyncMemory] = None

    async def _get_memory(self) -> AsyncMemory:
        """Lazily initialize and return the mem0 AsyncMemory instance."""
        if self._memory is None:
            self._memory = await AsyncMemory.from_config(config_dict=self._build_config())
            logger.info(
                "long_term_memory_initialized",
                collection_name=self._settings.LONG_TERM_MEMORY_COLLECTION_NAME,
            )
        return self._memory

    def _build_config(self) -> dict[str, Any]:
        """Build the mem0 configuration dictionary."""
        mem0_openai_config = build_mem0_openai_config(app_settings=self._settings, bifrost_agent="agent_1")
        config: dict[str, Any] = {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "collection_name": self._settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                    "dbname": self._settings.POSTGRES_DB,
                    "user": self._settings.POSTGRES_USER,
                    "password": self._settings.POSTGRES_PASSWORD,
                    "host": self._settings.POSTGRES_HOST,
                    "port": self._settings.POSTGRES_PORT,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._settings.LONG_TERM_MEMORY_MODEL,
                    **mem0_openai_config,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self._settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
                    **mem0_openai_config,
                },
            },
        }

        if self._settings.LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS:
            config["custom_instructions"] = self._settings.LONG_TERM_MEMORY_CUSTOM_INSTRUCTIONS

        return config

    async def search(self, user_id: int, query: str) -> str:
        """Get relevant memories for a user and query.

        Args:
            user_id: The user ID to search memories for.
            query: The query to search for relevant memories.

        Returns:
            Formatted string of relevant memories, or empty string on error.
        """
        try:
            memory = await self._get_memory()
            results = await memory.search(user_id=str(user_id), query=query)
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
        try:
            memory = await self._get_memory()
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
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
        asyncio.create_task(self.add(user_id, messages, metadata))


