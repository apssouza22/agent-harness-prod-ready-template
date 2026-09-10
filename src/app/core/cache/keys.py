import hashlib
import json

from src.app.core.cache.schemas import CacheableRequest


def build_params_data(request: CacheableRequest) -> dict:
    """Build request parameters used for cache scoping (excluding query text)."""
    return request.cache_scope()


def build_params_hash(request: CacheableRequest) -> str:
    """Hash request parameters so semantically similar queries share scope."""
    key_string = json.dumps(build_params_data(request), sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def build_exact_cache_key(request: CacheableRequest, prefix: str) -> str:
    """Generate exact cache key based on full request fingerprint."""
    key_data = {"query": request.cache_query, **build_params_data(request)}
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
    return f"{prefix}:exact:{key_hash}"
