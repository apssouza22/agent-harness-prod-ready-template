"""Parallel entity link index for memory search boosting."""

from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.db.connection_pool import get_connection_pool
from src.app.core.memory.entities import MemoryEntity


class EntityLinkStore:
    """Store and query entity-to-memory links scoped by user."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._table_name = f"{app_settings.LONG_TERM_MEMORY_COLLECTION_NAME}_entity_links"
        self._initialized = False

    async def _get_pool(self) -> AsyncConnectionPool | None:
        return await get_connection_pool(self._settings)

    async def initialize(self) -> None:
        """Create the entity link table and indexes if needed."""
        if self._initialized:
            return

        pool = await self._get_pool()
        if pool is None:
            logger.warning("entity_link_store_unavailable", reason="connection_pool_missing")
            return

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_id UUID NOT NULL,
                        entity_name TEXT NOT NULL,
                        entity_normalized TEXT NOT NULL,
                        entity_type TEXT NOT NULL DEFAULT 'unknown',
                        UNIQUE (user_id, memory_id, entity_normalized)
                    )
                    """
                )
                await cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._table_name}_user_entity_idx
                    ON {self._table_name} (user_id, entity_normalized)
                    """
                )
                await cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._table_name}_memory_idx
                    ON {self._table_name} (memory_id)
                    """
                )

        self._initialized = True
        logger.info("entity_link_store_initialized", table_name=self._table_name)

    async def replace_for_memory(self, user_id: str, memory_id: str, entities: list[MemoryEntity]) -> None:
        """Replace all entity links for a memory row."""
        pool = await self._get_pool()
        if pool is None:
            return

        await self.delete_for_memory(memory_id)
        if not entities:
            return

        rows = [
            (user_id, UUID(memory_id), entity.name, entity.normalized, entity.entity_type)
            for entity in entities
        ]
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    f"""
                    INSERT INTO {self._table_name}
                        (user_id, memory_id, entity_name, entity_normalized, entity_type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, memory_id, entity_normalized) DO NOTHING
                    """,
                    rows,
                )

    async def delete_for_memory(self, memory_id: str) -> None:
        """Remove all entity links for a deleted or replaced memory."""
        pool = await self._get_pool()
        if pool is None:
            return

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self._table_name} WHERE memory_id = %s",
                    (UUID(memory_id),),
                )

    async def list_user_entities(self, user_id: str) -> list[str]:
        """Return normalized entity names known for a user."""
        pool = await self._get_pool()
        if pool is None:
            return []

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT DISTINCT entity_normalized
                    FROM {self._table_name}
                    WHERE user_id = %s
                    ORDER BY entity_normalized
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()

        return [str(row[0]) for row in rows if row[0]]

    async def get_entities_for_memories(self, user_id: str, memory_ids: list[str]) -> dict[str, set[str]]:
        """Return normalized entities linked to each memory id."""
        if not memory_ids:
            return {}

        pool = await self._get_pool()
        if pool is None:
            return {}

        uuid_ids = [UUID(memory_id) for memory_id in memory_ids]
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT memory_id, entity_normalized
                    FROM {self._table_name}
                    WHERE user_id = %s AND memory_id = ANY(%s)
                    """,
                    (user_id, uuid_ids),
                )
                rows = await cur.fetchall()

        grouped: dict[str, set[str]] = {}
        for memory_id, entity_normalized in rows:
            key = str(memory_id)
            grouped.setdefault(key, set()).add(str(entity_normalized))
        return grouped
