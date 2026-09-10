from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class CacheConfidenceBreakdown:
    """Combined cache confidence scores for a candidate entry."""

    confidence: float
    exact_score: float
    fuzzy_score: float
    semantic_score: float
    matched_query: str


def fuzzy_ratio(query: str, cached_query: str) -> float:
    """Case-insensitive string similarity using SequenceMatcher."""
    return SequenceMatcher(None, query.lower(), cached_query.lower()).ratio()


def compute_confidence(
    query: str,
    cached_query: str,
    semantic_similarity: float,
    *,
    weight_exact: float,
    weight_fuzzy: float,
    weight_semantic: float,
) -> CacheConfidenceBreakdown:
    """Combine exact, fuzzy, and semantic scores into one confidence value."""
    exact_score = 1.0 if query.strip() == cached_query.strip() else 0.0
    fuzzy_score = fuzzy_ratio(query, cached_query)
    semantic_score = max(0.0, min(1.0, semantic_similarity))

    confidence = (
        weight_exact * exact_score + weight_fuzzy * fuzzy_score + weight_semantic * semantic_score
    )

    return CacheConfidenceBreakdown(
        confidence=confidence,
        exact_score=exact_score,
        fuzzy_score=fuzzy_score,
        semantic_score=semantic_score,
        matched_query=cached_query,
    )
