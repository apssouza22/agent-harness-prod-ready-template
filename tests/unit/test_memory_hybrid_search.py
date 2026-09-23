"""Unit tests for hybrid memory search fusion."""

from src.app.core.memory.search_ranker import fuse_hybrid_search_results
from src.app.core.memory.vector_store import MemoryRecord, PgVectorMemoryStore


def test_build_search_document_includes_entities() -> None:
    payload = {
        "data": "Works at Acme Corp",
        "entities": [{"name": "Acme Corp", "type": "organization"}, {"name": "Python", "type": "tool"}],
    }

    document = PgVectorMemoryStore.build_search_document(payload)

    assert "Works at Acme Corp" in document
    assert "Acme Corp" in document
    assert "Python" in document


def test_fuse_hybrid_search_promotes_keyword_only_hit() -> None:
    vector_records = [
        MemoryRecord(id="1", memory="Uses Rust daily", score=0.55, payload={}),
    ]
    keyword_records = [
        MemoryRecord(id="2", memory="Project Atlas uses PostgreSQL", score=0.92, payload={}),
    ]

    ranked = fuse_hybrid_search_results(
        vector_records,
        keyword_records,
        set(),
        {},
        vector_weight=0.55,
        keyword_weight=0.30,
        entity_weight=0.15,
        rrf_k=60,
        limit=1,
    )

    assert ranked[0].id == "2"


def test_fuse_hybrid_search_combines_vector_entity_and_keyword() -> None:
    vector_records = [
        MemoryRecord(id="1", memory="Uses Rust daily", score=0.10, payload={}),
        MemoryRecord(
            id="2",
            memory="Works at Acme Corp",
            score=0.20,
            payload={"entities": [{"name": "Acme Corp", "type": "organization"}]},
        ),
    ]
    keyword_records = [
        MemoryRecord(id="2", memory="Works at Acme Corp", score=0.80, payload={}),
    ]

    ranked = fuse_hybrid_search_results(
        vector_records,
        keyword_records,
        {"acme corp"},
        {"2": {"acme corp"}},
        vector_weight=0.55,
        keyword_weight=0.30,
        entity_weight=0.15,
        rrf_k=60,
        limit=1,
    )

    assert ranked[0].id == "2"
