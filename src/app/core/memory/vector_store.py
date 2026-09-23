"""Async pgvector storage for long-term memory records."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.db.connection_pool import get_connection_pool


@dataclass(frozen=True)
class MemoryRecord:
    """A stored memory row returned from vector search."""

    id: str
    memory: str
    score: float
    payload: dict[str, Any]


class PgVectorMemoryStore:
    """Persist and retrieve memory vectors in PostgreSQL with pgvector."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._collection_name = app_settings.LONG_TERM_MEMORY_COLLECTION_NAME
        self._dimensions = app_settings.LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS
        self._initialized = False

    @staticmethod
    def _format_vector(values: list[float]) -> str:
        return "[" + ",".join(str(value) for value in values) + "]"

    async def _get_pool(self) -> AsyncConnectionPool | None:
        return await get_connection_pool(self._settings)

    async def initialize(self) -> None:
        """Ensure pgvector extension, table, and index exist."""
        if self._initialized:
            return

        pool = await self._get_pool()
        if pool is None:
            logger.warning("memory_vector_store_unavailable", reason="connection_pool_missing")
            return

        table_name = self._collection_name
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id UUID PRIMARY KEY,
                        vector vector({self._dimensions}),
                        payload JSONB NOT NULL
                    )
                    """
                )
                await cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx
                    ON {table_name}
                    USING hnsw (vector vector_cosine_ops)
                    """
                )

        self._initialized = True
        logger.info("memory_vector_store_initialized", collection_name=table_name)

    async def insert(self, memory_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        """Insert a new memory vector."""
        pool = await self._get_pool()
        if pool is None:
            raise RuntimeError("memory_connection_pool_unavailable")

        table_name = self._collection_name
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {table_name} (id, vector, payload)
                    VALUES (%s, %s::vector, %s)
                    """,
                    (UUID(memory_id), self._format_vector(vector), Json(payload)),
                )

    async def search(
        self,
        vector: list[float],
        *,
        filters: dict[str, str],
        limit: int,
    ) -> list[MemoryRecord]:
        """Search memories by cosine distance with JSON payload filters."""
        pool = await self._get_pool()
        if pool is None:
            return []

        filter_conditions: list[str] = []
        filter_params: list[Any] = []
        for key, value in filters.items():
            filter_conditions.append("payload->>%s = %s")
            filter_params.extend([key, str(value)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""
        table_name = self._collection_name

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT id, vector <=> %s::vector AS distance, payload
                    FROM {table_name}
                    {filter_clause}
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (self._format_vector(vector), *filter_params, limit),
                )
                rows = await cur.fetchall()

        results: list[MemoryRecord] = []
        for row in rows:
            payload = row[2] if isinstance(row[2], dict) else {}
            results.append(
                MemoryRecord(
                    id=str(row[0]),
                    memory=str(payload.get("data", "")),
                    score=float(row[1]),
                    payload=payload,
                )
            )
        return results

    async def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single memory row by id."""
        pool = await self._get_pool()
        if pool is None:
            return None

        table_name = self._collection_name
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id, vector, payload FROM {table_name} WHERE id = %s",
                    (UUID(memory_id),),
                )
                row = await cur.fetchone()

        if not row:
            return None

        payload = row[2] if isinstance(row[2], dict) else {}
        return MemoryRecord(
            id=str(row[0]),
            memory=str(payload.get("data", "")),
            score=0.0,
            payload=payload,
        )

    async def update(self, memory_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        """Update an existing memory vector and payload."""
        pool = await self._get_pool()
        if pool is None:
            raise RuntimeError("memory_connection_pool_unavailable")

        table_name = self._collection_name
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {table_name}
                    SET vector = %s::vector, payload = %s
                    WHERE id = %s
                    """,
                    (self._format_vector(vector), Json(payload), UUID(memory_id)),
                )

    async def delete(self, memory_id: str) -> None:
        """Delete a memory row by id."""
        pool = await self._get_pool()
        if pool is None:
            raise RuntimeError("memory_connection_pool_unavailable")

        table_name = self._collection_name
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {table_name} WHERE id = %s",
                    (UUID(memory_id),),
                )
