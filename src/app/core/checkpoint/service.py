"""LangGraph checkpoint persistence and session cleanup."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.app.core.common.config import Environment, Settings, settings as default_settings
from src.app.core.common.logging import logger


class CheckpointService:
    """Service for LangGraph checkpoint persistence and session cleanup.

    Encapsulates AsyncPostgresSaver initialization and raw SQL cleanup of
    checkpoint tables for a given thread/session id.
    """

    def __init__(
        self,
        app_settings: Settings,
        connection_pool: AsyncConnectionPool | None,
    ) -> None:
        self._settings = app_settings
        self._connection_pool = connection_pool
        self._checkpointer: AsyncPostgresSaver | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def connection_pool(self) -> AsyncConnectionPool | None:
        return self._connection_pool

    async def get_checkpointer(self) -> AsyncPostgresSaver | None:
        """Return the LangGraph checkpointer, initializing it on first access."""
        if self._checkpointer is not None:
            return self._checkpointer

        if self._connection_pool is None:
            if self._settings.ENVIRONMENT != Environment.PRODUCTION:
                raise RuntimeError("connection pool initialization failed")
            return None

        self._checkpointer = AsyncPostgresSaver(self._connection_pool)
        await self._checkpointer.setup()
        logger.info(
            "checkpointer_initialized",
            environment=self._settings.ENVIRONMENT.value,
        )
        return self._checkpointer

    async def clear_session(self, session_id: str) -> None:
        """Clear all checkpoint rows for a session/thread id.

        Args:
            session_id: LangGraph thread id tied to the user session.

        Raises:
            RuntimeError: When the connection pool is unavailable.
            Exception: When a checkpoint table delete fails.
        """
        if self._connection_pool is None:
            logger.error("failed_to_clear_chat_history", error="connection pool unavailable")
            raise RuntimeError("connection pool unavailable")

        try:
            async with self._connection_pool.connection() as conn:
                for table in self._settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info("checkpoint_table_cleared", table=table, session_id=session_id)
                    except Exception as e:
                        logger.error("checkpoint_table_clear_failed", table=table, error=str(e))
                        raise
        except Exception as e:
            logger.error("failed_to_clear_chat_history", error=str(e))
            raise
