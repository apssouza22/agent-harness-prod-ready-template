"""Rank and fuse multi-signal memory search results."""

from src.app.core.memory.entities import compute_boosted_distance, compute_entity_overlap
from src.app.core.memory.vector_store import MemoryRecord


def _normalize_vector_similarity(distance: float) -> float:
    return max(0.0, 1.0 - distance)


def _normalize_keyword_scores(keyword_records: list[MemoryRecord]) -> dict[str, float]:
    if not keyword_records:
        return {}
    max_score = max(record.score for record in keyword_records)
    if max_score <= 0:
        return {record.id: 0.0 for record in keyword_records}
    return {record.id: record.score / max_score for record in keyword_records}


def fuse_hybrid_search_results(
    vector_records: list[MemoryRecord],
    keyword_records: list[MemoryRecord],
    query_entities: set[str],
    memory_entities: dict[str, set[str]],
    *,
    vector_weight: float,
    keyword_weight: float,
    entity_weight: float,
    rrf_k: int,
    limit: int,
) -> list[MemoryRecord]:
    """Fuse vector, keyword/BM25-like, and entity signals with weighted normalized scores."""
    del rrf_k  # retained for config compatibility

    if not vector_records and not keyword_records:
        return []

    records_by_id: dict[str, MemoryRecord] = {}
    for record in vector_records + keyword_records:
        records_by_id[record.id] = record

    vector_similarities = {record.id: _normalize_vector_similarity(record.score) for record in vector_records}
    keyword_similarities = _normalize_keyword_scores(keyword_records)

    scored: list[tuple[float, MemoryRecord]] = []
    for memory_id, record in records_by_id.items():
        linked_entities = memory_entities.get(memory_id, set())
        entity_overlap = compute_entity_overlap(query_entities, linked_entities)
        fused_score = (
            (vector_weight * vector_similarities.get(memory_id, 0.0))
            + (keyword_weight * keyword_similarities.get(memory_id, 0.0))
            + (entity_weight * entity_overlap)
        )
        scored.append(
            (
                fused_score,
                MemoryRecord(
                    id=record.id,
                    memory=record.memory,
                    score=fused_score,
                    payload=record.payload,
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:limit]]


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
