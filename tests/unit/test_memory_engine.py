"""Unit tests for LongTermMemoryEngine action execution."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.common.config import settings
from src.app.core.memory.engine import LongTermMemoryEngine
from src.app.core.memory.reconciler import MemoryAction
from src.app.core.memory.vector_store import MemoryRecord


@pytest.fixture
def engine() -> LongTermMemoryEngine:
    engine = LongTermMemoryEngine(settings, chat_model=MagicMock())
    engine._initialized = True
    engine._store = AsyncMock()
    engine._entity_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._extractor = AsyncMock()
    engine._entity_extractor = AsyncMock()
    engine._reconciler = AsyncMock()
    engine._embedder.embed.return_value = [0.1, 0.2, 0.3]
    engine._entity_extractor.extract_entities.return_value = []
    return engine


@pytest.mark.asyncio
async def test_add_applies_update_action(engine: LongTermMemoryEngine) -> None:
    engine._extractor.extract_facts.return_value = ["Works at Acme Corp"]
    engine._store.search_vector.return_value = [
        MemoryRecord(id="mem-1", memory="Works at Old Corp", score=0.05, payload={"data": "Works at Old Corp"})
    ]
    engine._reconciler.reconcile.return_value = [
        MemoryAction(
            event="UPDATE",
            text="Works at Acme Corp",
            memory_id="mem-1",
            previous_memory="Works at Old Corp",
        )
    ]
    engine._store.get.return_value = MemoryRecord(
        id="mem-1",
        memory="Works at Old Corp",
        score=0.0,
        payload={"data": "Works at Old Corp", "created_at": "2026-01-01T00:00:00+00:00"},
    )

    result = await engine.add(
        [{"role": "user", "content": "I now work at Acme Corp"}],
        user_id="42",
        metadata={"session_id": "s1"},
    )

    engine._store.update.assert_awaited_once()
    assert result["results"][0]["event"] == "UPDATE"
    assert result["results"][0]["memory"] == "Works at Acme Corp"


@pytest.mark.asyncio
async def test_add_applies_delete_action(engine: LongTermMemoryEngine) -> None:
    engine._extractor.extract_facts.return_value = ["Dislikes cheese pizza"]
    engine._store.search_vector.return_value = [
        MemoryRecord(id="mem-2", memory="Loves cheese pizza", score=0.04, payload={"data": "Loves cheese pizza"})
    ]
    engine._reconciler.reconcile.return_value = [
        MemoryAction(event="DELETE", text="Loves cheese pizza", memory_id="mem-2")
    ]

    result = await engine.add(
        [{"role": "user", "content": "I dislike cheese pizza now"}],
        user_id="42",
    )

    engine._store.delete.assert_awaited_once_with("mem-2")
    assert result["results"][0]["event"] == "DELETE"


@pytest.mark.asyncio
async def test_add_falls_back_to_add_when_reconcile_returns_empty(engine: LongTermMemoryEngine) -> None:
    engine._extractor.extract_facts.return_value = ["Uses Vim"]
    engine._store.search_vector.return_value = []
    engine._reconciler.reconcile.return_value = []

    result = await engine.add(
        [{"role": "user", "content": "I use Vim"}],
        user_id="42",
    )

    engine._store.insert.assert_awaited_once()
    assert result["results"][0]["event"] == "ADD"
    assert result["results"][0]["memory"] == "Uses Vim"


@pytest.mark.asyncio
async def test_search_applies_hybrid_entity_boost(engine: LongTermMemoryEngine) -> None:
    engine._settings.LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED = True
    engine._settings.LONG_TERM_MEMORY_KEYWORD_SEARCH_ENABLED = True
    engine._settings.LONG_TERM_MEMORY_SEARCH_LIMIT = 1
    engine._settings.LONG_TERM_MEMORY_ENTITY_SEARCH_POOL_MULTIPLIER = 2
    engine._settings.LONG_TERM_MEMORY_HYBRID_VECTOR_WEIGHT = 0.55
    engine._settings.LONG_TERM_MEMORY_HYBRID_KEYWORD_WEIGHT = 0.30
    engine._settings.LONG_TERM_MEMORY_HYBRID_ENTITY_WEIGHT = 0.15
    engine._store.search_vector.return_value = [
        MemoryRecord(id="1", memory="Uses Rust daily", score=0.10, payload={}),
        MemoryRecord(
            id="2",
            memory="Works at Acme Corp",
            score=0.20,
            payload={"entities": [{"name": "Acme Corp", "type": "organization"}]},
        ),
    ]
    engine._store.search_keyword.return_value = [
        MemoryRecord(id="2", memory="Works at Acme Corp", score=0.80, payload={}),
    ]
    engine._entity_store.list_user_entities.return_value = ["acme corp", "rust"]
    engine._entity_store.get_entities_for_memories.return_value = {"2": {"acme corp"}}

    result = await engine.search(user_id="42", query="Tell me about Acme Corp")

    engine._store.search_vector.assert_awaited_once()
    engine._store.search_keyword.assert_awaited_once()
    assert result["results"][0]["id"] == "2"
    assert result["results"][0]["memory"] == "Works at Acme Corp"
