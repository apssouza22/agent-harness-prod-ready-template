"""Database checkpointing and graph compilation utilities.

This module provides functions for managing PostgreSQL connection pooling,
graph compilation, and checkpoint management for the LangGraph agent.
"""

from src.app.core.checkpoint.factory import make_checkpointer, make_connection_pool
from src.app.core.common.config import settings
from src.app.core.common.logging import logger


async def get_checkpointer():
    """Backward-compatible wrapper around make_checkpointer."""
    return await make_checkpointer()


async def clear_checkpoints(session_id: str) -> None:
    """Clear all checkpoints for a session from database.

    Args:
        session_id: The session ID to clear checkpoints for.

    Raises:
        Exception: If there's an error clearing the checkpoints.
    """
    try:
        conn_pool = await make_connection_pool()

        async with conn_pool.connection() as conn:
            for table in settings.CHECKPOINT_TABLES:
                try:
                    await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                    logger.info("checkpoint_table_cleared", table=table, session_id=session_id)
                except Exception as e:
                    logger.error("checkpoint_table_clear_failed", table=table, error=str(e))
                    raise

    except Exception as e:
        logger.error("failed_to_clear_chat_history", error=str(e))
        raise
