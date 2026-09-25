from langchain_openai import OpenAIEmbeddings

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.llm.factory import build_openai_embeddings_kwargs


class OpenAIEmbeddingsClient:
    """Thin adapter wrapping LangChain OpenAI embeddings for cache lookups."""

    def __init__(self, settings: Settings):
        self.settings = settings
        kwargs = build_openai_embeddings_kwargs(
            settings,
            bifrost_agent="agent_1",
            model=settings.CACHE_EMBEDDER_MODEL,
            dimensions=settings.CACHE_EMBEDDING_DIMENSIONS,
        )
        self._embeddings = OpenAIEmbeddings(**kwargs)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string asynchronously."""
        try:
            vector = await self._embeddings.aembed_query(text)
            return list(vector)
        except Exception:
            logger.exception("openai_embeddings_failed")
            raise
