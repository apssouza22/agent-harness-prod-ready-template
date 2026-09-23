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
        self._keyword_config = app_settings.LONG_TERM_MEMORY_KEYWORD_SEARCH_CONFIG
        self._initialized = False

    @staticmethod
    def _format_vector(values: list[float]) -> str:
        return "[" + ",".join(str(value) for value in values) + "]"

    @staticmethod
    def build_search_document(payload: dict[str, Any]) -> str:
        """Build the text document indexed for keyword/BM25-like search."""
        parts = [str(payload.get("data", "")).strip()]
        entities = payload.get("entities", [])
        if isinstance(entities, list):
            for item in entities:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if name:
                        parts.append(name)
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
        return " ".join(part for part in parts if part)

    async def _get_pool(self) -> AsyncConnectionPool | None:
        return await get_connection_pool(self._settings)

    async def initialize(self) -> None:
        """Ensure pgvector extension, table, and indexes exist."""
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
                        payload JSONB NOT NULL,
                        search_text TSVECTOR
                    )
                    """
                )
                await cur.execute(
                    f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS search_text TSVECTOR
                    """
                )
                await cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx
                    ON {table_name}
                    USING hnsw (vector vector_cosine_ops)
                    """
                )
                await cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_search_text_idx
                    ON {table_name}
                    USING GIN (search_text)
                    """
                )
                await cur.execute(
                    f"""
                    UPDATE {table_name}
                    SET search_text = to_tsvector(%s, coalesce(payload->>'data', ''))
                    WHERE search_text IS NULL
                    """,
                    (self._keyword_config,),
                )

        self._initialized = True
        logger.info("memory_vector_store_initialized", collection_name=table_name)

    async def insert(self, memory_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        """Insert a new memory vector."""
        pool = await self._get_pool()
        if pool is None:
            raise RuntimeError("memory_connection_pool_unavailable")

        table_name = self._collection_name
        search_document = self.build_search_document(payload)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {table_name} (id, vector, payload, search_text)
                    VALUES (%s, %s::vector, %s, to_tsvector(%s, %s))
                    """,
                    (
                        UUID(memory_id),
                        self._format_vector(vector),
                        Json(payload),
                        self._keyword_config,
                        search_document,
                    ),
                )

    async def search(
        self,
        vector: list[float],
        *,
        filters: dict[str, str],
        limit: int,
    ) -> list[MemoryRecord]:
        """Search memories by cosine distance with JSON payload filters."""
        return await self.search_vector(vector, filters=filters, limit=limit)

    async def search_vector(
        self,
        vector: list[float],
        *,
        filters: dict[str, str],
        limit: int,
    ) -> list[MemoryRecord]:
        """Search memories by pgvector cosine distance."""
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

        return self._rows_to_records(rows, score_index=1)

    async def search_keyword(
        self,
        query: str,
        *,
        filters: dict[str, str],
        limit: int,
    ) -> list[MemoryRecord]:
        """Search memories using PostgreSQL BM25-like ranking via ts_rank_cd."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        pool = await self._get_pool()
        if pool is None:
            return []

        filter_conditions: list[str] = ["search_text @@ websearch_to_tsquery(%s, %s)"]
        filter_params: list[Any] = [self._keyword_config, cleaned_query]
        for key, value in filters.items():
            filter_conditions.append("payload->>%s = %s")
            filter_params.extend([key, str(value)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions)
        table_name = self._collection_name

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT
                        id,
                        ts_rank_cd(search_text, websearch_to_tsquery(%s, %s)) AS rank,
                        payload
                    FROM {table_name}
                    {filter_clause}
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (
                        self._keyword_config,
                        cleaned_query,
                        *filter_params,
                        limit,
                    ),
                )
                rows = await cur.fetchall()

        return self._rows_to_records(rows, score_index=1)

    @staticmethod
    def _rows_to_records(rows: list[tuple[Any, ...]], *, score_index: int) -> list[MemoryRecord]:
        results: list[MemoryRecord] = []
        for row in rows:
            payload = row[2] if isinstance(row[2], dict) else {}
            results.append(
                MemoryRecord(
                    id=str(row[0]),
                    memory=str(payload.get("data", "")),
                    score=float(row[score_index]),
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
        search_document = self.build_search_document(payload)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {table_name}
                    SET vector = %s::vector,
                        payload = %s,
                        search_text = to_tsvector(%s, %s)
                    WHERE id = %s
                    """,
                    (
                        self._format_vector(vector),
                        Json(payload),
                        self._keyword_config,
                        search_document,
                        UUID(memory_id),
                    ),
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
