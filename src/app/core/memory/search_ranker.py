"""Rank memory search results using entity overlap boosts."""

from src.app.core.memory.entities import compute_boosted_distance
from src.app.core.memory.vector_store import MemoryRecord


def rank_records_with_entity_boost(
    records: list[MemoryRecord],
    query_entities: set[str],
    memory_entities: dict[str, set[str]],
    *,
    boost_weight: float,
    limit: int,
) -> list[MemoryRecord]:
    """Rerank vector hits by subtracting an entity overlap boost from distance."""
    if not records:
        return []

    scored: list[tuple[float, MemoryRecord]] = []
    for record in records:
        linked_entities = memory_entities.get(record.id, set())
        boosted_distance = compute_boosted_distance(
            record.score,
            query_entities,
            linked_entities,
            boost_weight=boost_weight,
        )
        scored.append((boosted_distance, record))

    scored.sort(key=lambda item: item[0])
    return [record for _, record in scored[:limit]]
