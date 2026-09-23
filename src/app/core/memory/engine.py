"""Local long-term memory engine with fact extraction and memory reconciliation."""

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.memory.embedder import MemoryEmbedder
from src.app.core.memory.extractor import FactExtractor
from src.app.core.memory.reconciler import MemoryAction, MemoryReconciler
from src.app.core.memory.vector_store import MemoryRecord, PgVectorMemoryStore


class LongTermMemoryEngine:
    """Extract facts, reconcile against existing memories, and persist changes in pgvector."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._store = PgVectorMemoryStore(app_settings)
        self._embedder = MemoryEmbedder(app_settings)
        self._extractor = FactExtractor(app_settings)
        self._reconciler = MemoryReconciler(app_settings)
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
        """Extract facts, reconcile with existing memories, and apply ADD/UPDATE/DELETE actions."""
        await self.initialize()

        facts = await self._extractor.extract_facts(messages)
        if not facts:
            logger.debug("memory_add_skipped", reason="no_facts_extracted", user_id=user_id)
            return {"results": []}

        base_metadata = dict(metadata or {})
        base_metadata["user_id"] = user_id

        candidate_memories, id_mapping = await self._collect_candidate_memories(user_id, facts)
        existing_for_prompt = [{"id": temp_id, "text": record.memory} for temp_id, record in candidate_memories.items()]
        actions = await self._reconciler.reconcile(
            existing_for_prompt,
            facts,
            id_mapping=id_mapping,
        )
        if not actions:
            logger.warning("memory_reconcile_fallback_to_add", user_id=user_id, fact_count=len(facts))
            actions = [MemoryAction(event="ADD", text=fact) for fact in facts]

        stored_results: list[dict[str, Any]] = []
        for action in actions:
            result = await self._apply_action(action, base_metadata=base_metadata)
            if result:
                stored_results.append(result)

        return {"results": stored_results}

    async def _collect_candidate_memories(
        self,
        user_id: str,
        facts: list[str],
    ) -> tuple[dict[str, MemoryRecord], dict[str, str]]:
        """Find similar existing memories for each fact and assign temporary ids for the LLM."""
        candidates: dict[str, MemoryRecord] = {}
        temp_index = 0

        for fact in facts:
            embedding = await self._embedder.embed(fact)
            similar = await self._store.search(
                embedding,
                filters={"user_id": user_id},
                limit=self._settings.LONG_TERM_MEMORY_RECONCILE_CANDIDATE_LIMIT,
            )
            for record in similar:
                if record.id not in {existing.id for existing in candidates.values()}:
                    candidates[str(temp_index)] = record
                    temp_index += 1

        id_mapping = {temp_id: record.id for temp_id, record in candidates.items()}
        return candidates, id_mapping

    async def _apply_action(
        self,
        action: MemoryAction,
        *,
        base_metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute a single reconciled memory action."""
        if action.event == "NONE":
            logger.debug("memory_action_noop", memory_id=action.memory_id)
            return None

        if action.event == "DELETE":
            if not action.memory_id:
                return None
            await self._store.delete(action.memory_id)
            logger.info("memory_deleted", memory_id=action.memory_id)
            return {
                "id": action.memory_id,
                "memory": action.text,
                "event": "DELETE",
            }

        if action.event == "UPDATE":
            if not action.memory_id:
                return None
            existing = await self._store.get(action.memory_id)
            if existing is None:
                logger.warning("memory_update_missing", memory_id=action.memory_id)
                return None

            embedding = await self._embedder.embed(action.text)
            payload = self._build_updated_payload(existing, action.text, base_metadata)
            await self._store.update(action.memory_id, embedding, payload)
            logger.info("memory_updated", memory_id=action.memory_id)
            return {
                "id": action.memory_id,
                "memory": action.text,
                "event": "UPDATE",
                "previous_memory": action.previous_memory or existing.memory,
            }

        embedding = await self._embedder.embed(action.text)
        memory_id = str(uuid4())
        payload = self._build_new_payload(action.text, base_metadata)
        await self._store.insert(memory_id, embedding, payload)
        logger.info("memory_added", memory_id=memory_id)
        return {
            "id": memory_id,
            "memory": action.text,
            "event": "ADD",
        }

    def _build_new_payload(self, text: str, base_metadata: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            **base_metadata,
            "data": text,
            "hash": hashlib.md5(text.encode()).hexdigest(),
            "created_at": now,
            "updated_at": now,
        }

    def _build_updated_payload(
        self,
        existing: MemoryRecord,
        text: str,
        base_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {**existing.payload, **base_metadata}
        payload["data"] = text
        payload["hash"] = hashlib.md5(text.encode()).hexdigest()
        payload["created_at"] = existing.payload.get("created_at", datetime.now(UTC).isoformat())
        payload["updated_at"] = datetime.now(UTC).isoformat()
        return payload
