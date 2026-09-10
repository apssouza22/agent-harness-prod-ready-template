from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest

from src.app.api.v1.dtos.chat import ChatResponse
from src.app.core.cache.client import CacheClient, ExactCacheClient
from src.app.core.cache.keys import build_exact_cache_key, build_params_hash
from src.app.core.cache.scoring import CacheConfidenceBreakdown
from src.app.core.cache.semantic import SemanticCacheClient, embedding_to_bytes
from src.app.core.common.config import Settings
from src.app.core.common.model.message import Message


@dataclass
class _TestCacheRequest:
    query: str
    scope: dict

    @property
    def cache_query(self) -> str:
        return self.query

    def cache_scope(self) -> dict:
        return self.scope


@pytest.fixture
def cache_request() -> _TestCacheRequest:
    return _TestCacheRequest(
        query="What are transformers?",
        scope={"model": "gpt-test", "agent_name": "chatbot"},
    )


@pytest.fixture
def cache_response() -> ChatResponse:
    return ChatResponse(
        messages=[Message(role="assistant", content="Transformers are neural network architectures.")],
        trace_id="trace-123",
    )


@pytest.fixture
def confidence_breakdown() -> CacheConfidenceBreakdown:
    return CacheConfidenceBreakdown(
        confidence=0.78,
        exact_score=0.0,
        fuzzy_score=0.72,
        semantic_score=0.85,
        matched_query="What are transformer models?",
    )


@pytest.fixture
def test_settings() -> Settings:
    return Settings()


class TestCacheKeys:
    def test_params_hash_ignores_query_text(self, cache_request: _TestCacheRequest, test_settings: Settings):
        other_request = _TestCacheRequest(query="Explain transformer architecture", scope=cache_request.scope)
        assert build_params_hash(cache_request) == build_params_hash(other_request)

    def test_exact_cache_key_changes_with_query(self, cache_request: _TestCacheRequest, test_settings: Settings):
        other_request = _TestCacheRequest(query="Explain transformer architecture", scope=cache_request.scope)
        prefix = test_settings.CACHE_KEY_PREFIX
        assert build_exact_cache_key(cache_request, prefix) != build_exact_cache_key(other_request, prefix)

    def test_exact_cache_key_changes_with_params(self, cache_request: _TestCacheRequest, test_settings: Settings):
        other_request = _TestCacheRequest(query=cache_request.query, scope={"model": "other", "agent_name": "chatbot"})
        prefix = test_settings.CACHE_KEY_PREFIX
        assert build_exact_cache_key(cache_request, prefix) != build_exact_cache_key(other_request, prefix)


class TestEmbeddingBytes:
    def test_round_trip(self):
        vector = [0.1, 0.2, 0.3]
        packed = embedding_to_bytes(vector)
        assert len(packed) == len(vector) * 4


class TestCacheClient:
    @pytest.mark.asyncio
    async def test_lookup_returns_exact_hit_without_embedding(self, cache_request, cache_response, test_settings):
        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.find_cached_response = AsyncMock(return_value=cache_response)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=None,
            embeddings_client=None,
            settings=test_settings,
        )

        result = await cache_client.lookup(cache_request, ChatResponse)

        assert result.response == cache_response
        assert result.hit_type == "exact"
        assert result.query_embedding is None

    @pytest.mark.asyncio
    async def test_lookup_uses_confidence_cache_after_exact_miss(
        self, cache_request, cache_response, confidence_breakdown, test_settings, monkeypatch
    ):
        monkeypatch.setattr(test_settings, "CACHE_SEMANTIC_ENABLED", True)

        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.find_cached_response = AsyncMock(return_value=None)

        semantic_cache = Mock(spec=SemanticCacheClient)
        semantic_cache.is_ready = True
        semantic_cache.find_cached_response = AsyncMock(return_value=(cache_response, confidence_breakdown))

        embeddings_client = Mock()
        embeddings_client.embed_query = AsyncMock(return_value=[0.1] * test_settings.CACHE_EMBEDDING_DIMENSIONS)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=embeddings_client,
            settings=test_settings,
        )

        result = await cache_client.lookup(cache_request, ChatResponse)

        assert result.response == cache_response
        assert result.hit_type == "confidence"
        assert result.confidence == confidence_breakdown
        assert result.query_embedding == [0.1] * test_settings.CACHE_EMBEDDING_DIMENSIONS
        embeddings_client.embed_query.assert_awaited_once_with(cache_request.cache_query)

    @pytest.mark.asyncio
    async def test_lookup_returns_embedding_on_confidence_miss(self, cache_request, test_settings, monkeypatch):
        monkeypatch.setattr(test_settings, "CACHE_SEMANTIC_ENABLED", True)

        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.find_cached_response = AsyncMock(return_value=None)

        semantic_cache = Mock(spec=SemanticCacheClient)
        semantic_cache.is_ready = True
        semantic_cache.find_cached_response = AsyncMock(return_value=(None, None))

        embeddings_client = Mock()
        embeddings_client.embed_query = AsyncMock(return_value=[0.2] * test_settings.CACHE_EMBEDDING_DIMENSIONS)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=embeddings_client,
            settings=test_settings,
        )

        result = await cache_client.lookup(cache_request, ChatResponse)

        assert result.response is None
        assert result.hit_type is None
        assert result.query_embedding == [0.2] * test_settings.CACHE_EMBEDDING_DIMENSIONS

    @pytest.mark.asyncio
    async def test_store_writes_to_both_layers(self, cache_request, cache_response, test_settings, monkeypatch):
        monkeypatch.setattr(test_settings, "CACHE_SEMANTIC_ENABLED", True)

        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.store_response = AsyncMock(return_value=True)

        semantic_cache = Mock(spec=SemanticCacheClient)
        semantic_cache.is_ready = True
        semantic_cache.store_response = AsyncMock(return_value=True)

        embedding = [0.3] * test_settings.CACHE_EMBEDDING_DIMENSIONS
        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=None,
            settings=test_settings,
        )

        await cache_client.store(cache_request, cache_response, query_embedding=embedding)

        exact_cache.store_response.assert_awaited_once_with(cache_request, cache_response)
        semantic_cache.store_response.assert_awaited_once_with(cache_request, cache_response, embedding)
