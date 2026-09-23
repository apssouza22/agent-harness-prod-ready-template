"""Unit tests for entity-linked memory search boosting."""

from src.app.core.memory.entities import (
    compute_boosted_distance,
    compute_entity_overlap,
    match_query_entities,
    normalize_entity_name,
)
from src.app.core.memory.search_ranker import rank_records_with_entity_boost
from src.app.core.memory.vector_store import MemoryRecord


def test_normalize_entity_name() -> None:
    assert normalize_entity_name("  Acme   Corp ") == "acme corp"


def test_match_query_entities_prefers_longest_matches() -> None:
    known = ["acme", "acme corp", "python"]
    matched = match_query_entities("What is Acme Corp using for Python?", known)

    assert matched == {"acme corp", "python"}


def test_compute_entity_overlap() -> None:
    overlap = compute_entity_overlap({"acme corp", "python"}, {"acme corp", "rust"})
    assert overlap == 0.5


def test_compute_boosted_distance_reduces_distance_on_overlap() -> None:
    boosted = compute_boosted_distance(
        0.30,
        {"acme corp"},
        {"acme corp"},
        boost_weight=0.15,
    )
    assert boosted == 0.15


def test_rank_records_with_entity_boost_promotes_matching_memory() -> None:
    records = [
        MemoryRecord(id="1", memory="Uses Rust daily", score=0.10, payload={}),
        MemoryRecord(
            id="2",
            memory="Works at Acme Corp",
            score=0.20,
            payload={"entities": [{"name": "Acme Corp", "type": "organization"}]},
        ),
    ]
    memory_entities = {"2": {"acme corp"}}

    ranked = rank_records_with_entity_boost(
        records,
        {"acme corp"},
        memory_entities,
        boost_weight=0.15,
        limit=1,
    )

    assert ranked[0].id == "2"
