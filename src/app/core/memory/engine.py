"""Local long-term memory engine using ADD-only fact extraction and pgvector retrieval."""

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.embedder import MemoryEmbedder
from src.app.core.memory.extractor import FactExtractor
from src.app.core.memory.vector_store import PgVectorMemoryStore


class LongTermMemoryEngine:
    """Extract facts, store vectors in pgvector, and retrieve by semantic similarity."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._store = PgVectorMemoryStore(app_settings)
        self._embedder = MemoryEmbedder(app_settings)
        self._extractor = FactExtractor(app_settings)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the vector store once per process."""
        if self._initialized:
            return
        await self._store.initialize()
        self._initialized = True

    async def search(self, user_id: str, query: str) -> dict[str, list[dict[str, Any]]]:
        """Search relevant memories for a user query."""
        await self.initialize()

        embedding = await self._embedder.embed(query)
        records = await self._store.search(
            embedding,
            filters={"user_id": user_id},
            limit=self._settings.LONG_TERM_MEMORY_SEARCH_LIMIT,
        )

        results = [
            {
                "id": record.id,
                "memory": record.memory,
                "score": record.score,
            }
            for record in records
            if record.memory
        ]
        return {"results": results}

    async def add(
        self,
        messages: list[dict[str, Any]],
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract facts from messages and append new memories for the user."""
        await self.initialize()

        facts = await self._extractor.extract_facts(messages)
        if not facts:
            logger.debug("memory_add_skipped", reason="no_facts_extracted", user_id=user_id)
            return {"results": []}

        base_metadata = dict(metadata or {})
        base_metadata["user_id"] = user_id
        stored_results: list[dict[str, Any]] = []

        for fact in facts:
            embedding = await self._embedder.embed(fact)
            similar = await self._store.search(
                embedding,
                filters={"user_id": user_id},
                limit=1,
            )
            if similar and similar[0].score <= self._settings.LONG_TERM_MEMORY_DEDUP_DISTANCE:
                logger.debug(
                    "memory_add_skipped_duplicate",
                    user_id=user_id,
                    distance=similar[0].score,
                )
                continue

            memory_id = str(uuid4())
            payload = {
                **base_metadata,
                "data": fact,
                "hash": hashlib.md5(fact.encode()).hexdigest(),
                "created_at": datetime.now(UTC).isoformat(),
            }
            await self._store.insert(memory_id, embedding, payload)
            stored_results.append(
                {
                    "id": memory_id,
                    "memory": fact,
                    "event": "ADD",
                }
            )

        return {"results": stored_results}
