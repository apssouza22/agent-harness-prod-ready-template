"""Cache client factory."""

import redis

from src.app.core.cache.client import CacheClient, EmbeddingsClient, ExactCacheClient
from src.app.core.cache.semantic import SemanticCacheClient
from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.common.logging import logger


def get_settings() -> Settings:
    """Return application settings (imported lazily for test overrides)."""
    return default_settings


def make_redis_client(settings: Settings, *, decode_responses: bool | None = None) -> redis.Redis:
    """Create Redis client with connection pooling."""
    decode = settings.REDIS_DECODE_RESPONSES if decode_responses is None else decode_responses

    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
        db=settings.REDIS_DB,
        decode_responses=decode,
        socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
        retry_on_timeout=True,
        retry_on_error=[redis.ConnectionError, redis.TimeoutError],
    )

    client.ping()
    logger.info(
        "redis_connected",
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=decode,
    )
    return client


def _make_semantic_cache_client(settings: Settings) -> SemanticCacheClient | None:
    if not settings.CACHE_SEMANTIC_ENABLED:
        logger.info("semantic_cache_disabled")
        return None

    try:
        semantic_redis = make_redis_client(settings, decode_responses=False)
        semantic_cache = SemanticCacheClient(semantic_redis, settings)
        logger.info("semantic_cache_client_created")
        return semantic_cache
    except Exception:
        logger.warning("semantic_cache_unavailable", exc_info=True)
        return None


def make_cache_client(
    settings: Settings,
    embeddings_client: EmbeddingsClient | None = None,
) -> CacheClient | None:
    """Create layered exact + semantic cache client."""
    try:
        exact_redis = make_redis_client(settings, decode_responses=True)
        exact_cache = ExactCacheClient(exact_redis, settings)
        semantic_cache = _make_semantic_cache_client(settings)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=embeddings_client,
            settings=settings,
        )
        logger.info("cache_client_created")
        return cache_client
    except Exception:
        logger.warning("cache_unavailable", exc_info=True)
        return None


def make_cache_client_fresh(
    settings: Settings,
    embeddings_client: EmbeddingsClient | None = None,
) -> CacheClient | None:
    """Create a new cache client instance (bypasses any future caching)."""
    return make_cache_client(settings, embeddings_client=embeddings_client)
