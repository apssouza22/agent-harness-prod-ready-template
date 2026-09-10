import struct
import uuid
from datetime import timedelta
from typing import List, Optional, TypeVar

import redis
from pydantic import BaseModel
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from src.app.core.cache.keys import build_params_hash
from src.app.core.cache.schemas import CacheableRequest
from src.app.core.cache.scoring import CacheConfidenceBreakdown, compute_confidence
from src.app.core.common.config import Settings
from src.app.core.common.logging import logger

TResponse = TypeVar("TResponse", bound=BaseModel)


def embedding_to_bytes(embedding: List[float]) -> bytes:
    """Pack a float embedding vector into Redis vector field bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _decode_field(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class SemanticCacheClient:
    """Redis Stack vector cache with confidence scoring (exact + fuzzy + semantic)."""

    def __init__(self, redis_client: redis.Redis, settings: Settings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.CACHE_TTL_HOURS)
        self.candidate_min_similarity = settings.CACHE_SEMANTIC_SIMILARITY_THRESHOLD
        self.confidence_threshold = settings.CACHE_CONFIDENCE_THRESHOLD
        self.search_top_k = settings.CACHE_SEMANTIC_SEARCH_TOP_K
        self.embedding_dimensions = settings.CACHE_EMBEDDING_DIMENSIONS
        self.index_name = f"{settings.CACHE_KEY_PREFIX}_semantic_cache_idx"
        self.key_prefix = f"{settings.CACHE_KEY_PREFIX}_semantic_cache:"
        self._index_ready = False
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it does not exist."""
        try:
            self.redis.ft(self.index_name).info()
            self._index_ready = True
            logger.info("semantic_cache_index_exists", index_name=self.index_name)
            return
        except ResponseError:
            pass

        schema = (
            TagField("params_hash"),
            TextField("query"),
            TextField("response"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dimensions,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH)

        try:
            self.redis.ft(self.index_name).create_index(schema, definition=definition)
            self._index_ready = True
            logger.info("semantic_cache_index_created", index_name=self.index_name)
        except ResponseError as e:
            if "Index already exists" in str(e):
                self._index_ready = True
                logger.info("semantic_cache_index_exists", index_name=self.index_name)
                return
            raise

    @property
    def is_ready(self) -> bool:
        return self._index_ready

    def _score_candidate(
        self,
        query: str,
        cached_query: str,
        semantic_similarity: float,
    ) -> CacheConfidenceBreakdown:
        return compute_confidence(
            query,
            cached_query,
            semantic_similarity,
            weight_exact=self.settings.CACHE_WEIGHT_EXACT,
            weight_fuzzy=self.settings.CACHE_WEIGHT_FUZZY,
            weight_semantic=self.settings.CACHE_WEIGHT_SEMANTIC,
        )

    def _select_best_candidate(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
    ) -> Optional[tuple[CacheConfidenceBreakdown, str]]:
        best_match: Optional[tuple[CacheConfidenceBreakdown, str]] = None

        for cached_query, response_json, semantic_similarity in candidates:
            if semantic_similarity < self.candidate_min_similarity:
                continue

            breakdown = self._score_candidate(query, cached_query, semantic_similarity)
            logger.debug(
                "semantic_cache_candidate_scored",
                matched_query=cached_query[:80],
                exact_score=breakdown.exact_score,
                fuzzy_score=breakdown.fuzzy_score,
                semantic_score=breakdown.semantic_score,
                confidence=breakdown.confidence,
            )

            if best_match is None or breakdown.confidence > best_match[0].confidence:
                best_match = (breakdown, response_json)

        if best_match and best_match[0].confidence >= self.confidence_threshold:
            return best_match

        return None

    async def find_cached_response(
        self,
        request: CacheableRequest,
        query_embedding: List[float],
        response_model: type[TResponse],
    ) -> tuple[Optional[TResponse], Optional[CacheConfidenceBreakdown]]:
        """Find a cached response using confidence-based fuzzy + semantic scoring."""
        if not self._index_ready:
            return None, None

        if len(query_embedding) != self.embedding_dimensions:
            logger.warning(
                "semantic_cache_embedding_dimension_mismatch",
                query_dimensions=len(query_embedding),
                expected_dimensions=self.embedding_dimensions,
            )
            return None, None

        try:
            params_hash = build_params_hash(request)
            vec_bytes = embedding_to_bytes(query_embedding)

            query = (
                Query(
                    f"(@params_hash:{{{params_hash}}})=>[KNN {self.search_top_k} @embedding $vec AS distance]"
                )
                .sort_by("distance")
                .return_fields("response", "distance", "query")
                .dialect(2)
            )

            results = self.redis.ft(self.index_name).search(query, query_params={"vec": vec_bytes})
            if not results.docs:
                return None, None

            candidates: list[tuple[str, str, float]] = []
            for doc in results.docs:
                distance = float(getattr(doc, "distance", 2.0))
                semantic_similarity = max(0.0, 1.0 - distance)
                cached_query = _decode_field(doc.query)
                response_json = _decode_field(doc.response)
                candidates.append((cached_query, response_json, semantic_similarity))

            best_match = self._select_best_candidate(request.cache_query, candidates)
            if not best_match:
                logger.debug("semantic_cache_miss", reason="confidence_threshold_not_met")
                return None, None

            best_breakdown, matched_response = best_match

            logger.info(
                "semantic_cache_hit",
                confidence=best_breakdown.confidence,
                exact_score=best_breakdown.exact_score,
                fuzzy_score=best_breakdown.fuzzy_score,
                semantic_score=best_breakdown.semantic_score,
                matched_query=best_breakdown.matched_query[:80],
            )
            return response_model.model_validate_json(matched_response), best_breakdown

        except Exception:
            logger.exception("semantic_cache_lookup_failed")
            return None, None

    async def store_response(
        self,
        request: CacheableRequest,
        response: BaseModel,
        query_embedding: List[float],
    ) -> bool:
        """Store a response for semantic similarity lookups."""
        if not self._index_ready:
            return False

        if len(query_embedding) != self.embedding_dimensions:
            logger.warning("semantic_cache_store_skipped", reason="embedding_dimension_mismatch")
            return False

        try:
            cache_key = f"{self.key_prefix}{uuid.uuid4().hex}"
            mapping = {
                "params_hash": build_params_hash(request),
                "query": request.cache_query,
                "response": response.model_dump_json(),
                "embedding": embedding_to_bytes(query_embedding),
            }

            pipe = self.redis.pipeline()
            pipe.hset(cache_key, mapping=mapping)
            pipe.expire(cache_key, int(self.ttl.total_seconds()))
            pipe.execute()

            logger.info("semantic_cache_stored", cache_key_prefix=cache_key[:24])
            return True

        except Exception:
            logger.exception("semantic_cache_store_failed")
            return False
