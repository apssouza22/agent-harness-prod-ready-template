"""Entity models and ranking helpers for memory search boosting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEntity:
    """An extracted entity linked to a stored memory."""

    name: str
    entity_type: str = "unknown"

    @property
    def normalized(self) -> str:
        return normalize_entity_name(self.name)


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for overlap comparisons."""
    return " ".join(name.strip().lower().split())


def match_query_entities(query: str, known_entities: list[str]) -> set[str]:
    """Match known user entities that appear in a query string."""
    normalized_query = query.lower()
    matched: set[str] = set()
    for entity in sorted(known_entities, key=len, reverse=True):
        if not entity or entity not in normalized_query:
            continue
        if any(entity in existing for existing in matched):
            continue
        matched.add(entity)
    return matched


def compute_entity_overlap(query_entities: set[str], memory_entities: set[str]) -> float:
    """Return the fraction of query entities that overlap with a memory."""
    if not query_entities or not memory_entities:
        return 0.0
    return len(query_entities & memory_entities) / len(query_entities)


def compute_boosted_distance(
    vector_distance: float,
    query_entities: set[str],
    memory_entities: set[str],
    *,
    boost_weight: float,
) -> float:
    """Reduce vector distance when entity overlap exists (lower distance is better)."""
    overlap = compute_entity_overlap(query_entities, memory_entities)
    return vector_distance - (overlap * boost_weight)
