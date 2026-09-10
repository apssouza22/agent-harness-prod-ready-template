import pytest

from src.app.core.cache.semantic import SemanticCacheClient
from src.app.core.common.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def semantic_client(settings: Settings) -> SemanticCacheClient:
    client = SemanticCacheClient.__new__(SemanticCacheClient)
    client.settings = settings
    client.candidate_min_similarity = settings.CACHE_SEMANTIC_SIMILARITY_THRESHOLD
    client.confidence_threshold = settings.CACHE_CONFIDENCE_THRESHOLD
    client.search_top_k = settings.CACHE_SEMANTIC_SEARCH_TOP_K
    client.embedding_dimensions = settings.CACHE_EMBEDDING_DIMENSIONS
    client._index_ready = True
    return client


class TestSemanticCandidateSelection:
    def test_selects_highest_confidence_candidate(self, semantic_client: SemanticCacheClient):
        candidates = [
            ("What are transformers?", "{}", 0.98),
            ("Explain transformer architecture in ML", "{}", 0.80),
            ("How does backpropagation work?", "{}", 0.75),
        ]

        best_match = semantic_client._select_best_candidate(
            "What are transformers?",
            candidates,
        )

        assert best_match is not None
        breakdown, _ = best_match
        assert breakdown.matched_query == "What are transformers?"
        assert breakdown.confidence >= 0.90

    def test_rejects_candidates_below_confidence_threshold(self, semantic_client: SemanticCacheClient):
        candidates = [
            ("Completely unrelated topic about databases", "{}", 0.71),
        ]

        best_match = semantic_client._select_best_candidate(
            "What are transformers?",
            candidates,
        )

        assert best_match is None

    def test_rejects_borderline_match_below_confidence_threshold(self, semantic_client: SemanticCacheClient):
        candidates = [
            ("What are transformers in machine learning?", "{}", 0.72),
        ]

        best_match = semantic_client._select_best_candidate(
            "What are transformers in ML?",
            candidates,
        )

        assert best_match is None
