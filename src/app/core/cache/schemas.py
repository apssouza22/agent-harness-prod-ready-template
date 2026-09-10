import json
from dataclasses import dataclass
from typing import Generic, Literal, Optional, Protocol, TypeVar

from pydantic import BaseModel

from src.app.core.cache.scoring import CacheConfidenceBreakdown
from src.app.core.common.model.message import Message

TResponse = TypeVar("TResponse", bound=BaseModel)


class CacheableRequest(Protocol):
    """Anything that can be cached must expose query text and scope params."""

    @property
    def cache_query(self) -> str:
        """Text fingerprint used for exact key, fuzzy match, and embedding."""
        ...

    def cache_scope(self) -> dict:
        """Parameters that change the answer (model, agent, etc.) — excludes query text."""
        ...


@dataclass
class CacheLookupResult(Generic[TResponse]):
    """Result of a layered cache lookup."""

    response: Optional[TResponse] = None
    query_embedding: Optional[list[float]] = None
    hit_type: Optional[Literal["exact", "confidence"]] = None
    confidence: Optional[CacheConfidenceBreakdown] = None


class ChatCacheRequest:
    """Adapter for chat endpoint cache lookups and stores.

    Design decisions:
    - Scope by model + agent_name only (not session_id) for cross-session reuse.
    - Serialize full message history as cache_query so multi-turn context is respected.
    - session_id is intentionally excluded from cache_scope; add user_id there if
      personalization must prevent cross-user reuse.
    """

    def __init__(
        self,
        messages: list[Message],
        *,
        model: str,
        agent_name: str,
        session_id: str,
    ):
        self.messages = messages
        self.model = model
        self.agent_name = agent_name
        self.session_id = session_id

    @property
    def cache_query(self) -> str:
        payload = [{"role": m.role, "content": m.content} for m in self.messages]
        return json.dumps(payload, sort_keys=True)

    def cache_scope(self) -> dict:
        return {
            "model": self.model,
            "agent_name": self.agent_name,
        }
