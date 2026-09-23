"""PostgreSQL JSONB persistence for session dialogue state."""

from datetime import UTC, datetime

from psycopg_pool import AsyncConnectionPool

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.db.connection_pool import get_connection_pool
from src.app.core.dialogue_state.models import DialogueState


class DialogueStateStore:
    """Store and load dialogue state rows scoped by session."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._table_name = app_settings.DIALOGUE_STATE_TABLE_NAME
        self._initialized = False

    async def _get_pool(self) -> AsyncConnectionPool | None:
        return await get_connection_pool(self._settings)

    async def initialize(self) -> None:
        """Create the dialogue state table if needed."""
        if self._initialized:
            return

        pool = await self._get_pool()
        if pool is None:
            logger.warning("dialogue_state_store_unavailable", reason="connection_pool_missing")
            return

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        state JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                await cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._table_name}_user_idx
                    ON {self._table_name} (user_id)
                    """
                )

        self._initialized = True
        logger.info("dialogue_state_store_initialized", table_name=self._table_name)

    async def get(self, session_id: str) -> DialogueState | None:
        """Load dialogue state for a session."""
        await self.initialize()

        pool = await self._get_pool()
        if pool is None:
            return None

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT state FROM {self._table_name} WHERE session_id = %s",
                    (session_id,),
                )
                row = await cur.fetchone()

        if row is None:
            return None

        return DialogueState.model_validate(row[0])

    async def upsert(self, session_id: str, user_id: str, state: DialogueState) -> None:
        """Persist dialogue state for a session."""
        await self.initialize()

        pool = await self._get_pool()
        if pool is None:
            return

        payload = state.model_dump()
        updated_at = datetime.now(UTC)

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {self._table_name} (session_id, user_id, state, updated_at)
                    VALUES (%s, %s, %s::jsonb, %s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        state = EXCLUDED.state,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (session_id, user_id, payload, updated_at),
                )
