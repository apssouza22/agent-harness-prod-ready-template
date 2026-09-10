from datetime import timedelta
from typing import Optional, Protocol, TypeVar

import redis
from pydantic import BaseModel

from src.app.core.cache.keys import build_exact_cache_key
from src.app.core.cache.schemas import CacheLookupResult, CacheableRequest
from src.app.core.cache.semantic import SemanticCacheClient
from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.metrics.metrics import cache_hits_total, cache_misses_total

TResponse = TypeVar("TResponse", bound=BaseModel)


class EmbeddingsClient(Protocol):
    """Protocol for embedding providers used by the semantic cache layer."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string for semantic cache lookup."""
        ...


class ExactCacheClient:
    """Redis-based exact match cache."""

    def __init__(self, redis_client: redis.Redis, settings: Settings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.CACHE_TTL_HOURS)
        self.key_prefix = settings.CACHE_KEY_PREFIX

    async def find_cached_response(
        self,
        request: CacheableRequest,
        response_model: type[TResponse],
    ) -> Optional[TResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = build_exact_cache_key(request, self.key_prefix)
            cached_response = self.redis.get(cache_key)

            if cached_response:
                try:
                    logger.info("exact_cache_hit", cache_key_prefix=cache_key[:24])
                    return response_model.model_validate_json(cached_response)
                except Exception:
                    logger.warning("exact_cache_deserialize_failed", cache_key_prefix=cache_key[:24])
                    return None

            return None

        except Exception:
            logger.exception("exact_cache_lookup_failed")
            return None

    async def store_response(self, request: CacheableRequest, response: BaseModel) -> bool:
        """Store response for exact query matching."""
        try:
            cache_key = build_exact_cache_key(request, self.key_prefix)
            success = self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            if success:
                logger.info("exact_cache_stored", cache_key_prefix=cache_key[:24])
                return True

            logger.warning("exact_cache_store_failed", cache_key_prefix=cache_key[:24])
            return False

        except Exception:
            logger.exception("exact_cache_store_failed")
            return False


class CacheClient:
    """Layered cache: exact match first, then confidence-based fuzzy + semantic matching."""

    def __init__(
        self,
        exact_cache: ExactCacheClient,
        semantic_cache: Optional[SemanticCacheClient],
        embeddings_client: Optional[EmbeddingsClient],
        settings: Settings,
    ):
        self.exact = exact_cache
        self.semantic = semantic_cache
        self.embeddings = embeddings_client
        self.settings = settings

    async def _embed_query(self, request: CacheableRequest) -> Optional[list[float]]:
        if not self.embeddings:
            return None

        try:
            return await self.embeddings.embed_query(request.cache_query)
        except Exception:
            logger.warning("cache_embedding_failed", exc_info=True)
            return None

    async def lookup(
        self,
        request: CacheableRequest,
        response_model: type[TResponse],
    ) -> CacheLookupResult[TResponse]:
        """Check exact cache, then semantic cache. Returns embedding for reuse on miss."""
        exact_hit = await self.exact.find_cached_response(request, response_model)
        if exact_hit:
            cache_hits_total.labels(layer="exact").inc()
            return CacheLookupResult(response=exact_hit, hit_type="exact")

        if not self.settings.CACHE_SEMANTIC_ENABLED or not self.semantic or not self.semantic.is_ready:
            cache_misses_total.inc()
            return CacheLookupResult()

        query_embedding = await self._embed_query(request)
        if not query_embedding:
            cache_misses_total.inc()
            return CacheLookupResult()

        semantic_hit, confidence_breakdown = await self.semantic.find_cached_response(
            request,
            query_embedding,
            response_model,
        )
        if semantic_hit:
            cache_hits_total.labels(layer="semantic").inc()
            return CacheLookupResult(
                response=semantic_hit,
                query_embedding=query_embedding,
                hit_type="confidence",
                confidence=confidence_breakdown,
            )

        cache_misses_total.inc()
        return CacheLookupResult(query_embedding=query_embedding)

    async def store(
        self,
        request: CacheableRequest,
        response: BaseModel,
        query_embedding: Optional[list[float]] = None,
    ) -> None:
        """Store response in exact and semantic caches."""
        await self.exact.store_response(request, response)

        if not self.settings.CACHE_SEMANTIC_ENABLED or not self.semantic or not self.semantic.is_ready:
            return

        embedding = query_embedding or await self._embed_query(request)
        if embedding:
            await self.semantic.store_response(request, response, embedding)
