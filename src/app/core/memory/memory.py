"""Long-term memory management using mem0 and pgvector.

This module provides functions for managing long-term memory operations including
initialization, search, and updates using the mem0 library with PostgreSQL/pgvector backend.
"""
import asyncio
from typing import Optional

from mem0 import AsyncMemory

from src.app.core.common.config import settings
from src.app.core.common.logging import logger

# Module-level singleton for memory instance
_memory_instance: Optional[AsyncMemory] = None


async def get_memory_instance() -> AsyncMemory:
    """Initialize and return the long-term memory singleton.

    Returns:
        AsyncMemory: The initialized memory instance.
    """
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = await AsyncMemory.from_config(
            config_dict={
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                        "dbname": settings.POSTGRES_DB,
                        "user": settings.POSTGRES_USER,
                        "password": settings.POSTGRES_PASSWORD,
                        "host": settings.POSTGRES_HOST,
                        "port": settings.POSTGRES_PORT,
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {"model": settings.LONG_TERM_MEMORY_MODEL},
                },
                "embedder": {"provider": "openai", "config": {"model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL}},
                # "custom_fact_extraction_prompt": load_custom_fact_extraction_prompt(),
            }
        )
    return _memory_instance


async def get_relevant_memory(user_id: int, query: str) -> str:
    """Get relevant memories for user and query.

    Args:
        user_id: The user ID to search memories for.
        query: The query to search for relevant memories.

    Returns:
        str: Formatted string of relevant memories, or empty string on error.
    """
    try:
        memory = await get_memory_instance()
        results = await memory.search(user_id=str(user_id), query=query)
        print(results)
        return "\n".join([f"* {result['memory']}" for result in results["results"]])
    except Exception as e:
        logger.error("failed_to_get_relevant_memory", error=str(e), user_id=user_id, query=query)
        return ""


async def update_memory(user_id: int, messages: list[dict], metadata: dict = None) -> None:
    """Update long-term memory with new messages.

    Args:
        user_id: The user ID to update memory for.
        messages: The messages to add to memory.
        metadata: Optional metadata to include with the memory update.
    """
    try:
        memory = await get_memory_instance()
        await memory.add(messages, user_id=str(user_id), metadata=metadata)
        logger.info("long_term_memory_updated_successfully", user_id=user_id)
    except Exception as e:
        logger.exception(
            "failed_to_update_long_term_memory",
            user_id=user_id,
            error=str(e),
        )

def bg_update_memory(user_id: int, messages: list[dict], metadata: dict = None) -> None:
    """Run memory update in background without blocking the response

    Args:
        user_id: The user ID to update memory for.
        messages: The messages to add to memory.
        metadata: Optional metadata to include with the memory update.
    """
    asyncio.create_task(
        update_memory(user_id, messages, metadata)
    )

