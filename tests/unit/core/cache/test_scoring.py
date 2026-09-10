import pytest

from src.app.core.cache.scoring import compute_confidence, fuzzy_ratio


class TestFuzzyRatio:
    def test_identical_strings(self):
        assert fuzzy_ratio("What are transformers?", "What are transformers?") == 1.0

    def test_case_insensitive(self):
        assert fuzzy_ratio("Hello World", "hello world") == 1.0

    def test_similar_strings(self):
        ratio = fuzzy_ratio("What are transformers?", "What are transformer models?")
        assert 0.5 < ratio < 1.0

    def test_different_strings(self):
        ratio = fuzzy_ratio("What are transformers?", "How does gradient descent work?")
        assert ratio < 0.5


class TestComputeConfidence:
    def test_exact_match_scores_highest(self):
        breakdown = compute_confidence(
            "What are transformers?",
            "What are transformers?",
            semantic_similarity=0.95,
            weight_exact=0.10,
            weight_fuzzy=0.20,
            weight_semantic=0.70,
        )

        assert breakdown.exact_score == 1.0
        assert breakdown.fuzzy_score == 1.0
        assert breakdown.confidence == pytest.approx(0.965)

    def test_weighted_confidence_matches_reference_formula(self):
        breakdown = compute_confidence(
            "Explain transformer architecture",
            "What are transformers in ML?",
            semantic_similarity=0.82,
            weight_exact=0.10,
            weight_fuzzy=0.20,
            weight_semantic=0.70,
        )

        expected = (
            0.10 * breakdown.exact_score
            + 0.20 * breakdown.fuzzy_score
            + 0.70 * breakdown.semantic_score
        )
        assert breakdown.confidence == pytest.approx(expected)
        assert breakdown.exact_score == 0.0
        assert breakdown.semantic_score == 0.82

    def test_confidence_threshold_example(self):
        breakdown = compute_confidence(
            "What are transformers?",
            "What are transformers?",
            semantic_similarity=0.95,
            weight_exact=0.10,
            weight_fuzzy=0.20,
            weight_semantic=0.70,
        )

        assert breakdown.confidence >= 0.90
