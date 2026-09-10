from langchain_openai import OpenAIEmbeddings

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger


class OpenAIEmbeddingsClient:
    """Thin adapter wrapping LangChain OpenAI embeddings for cache lookups."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.CACHE_EMBEDDER_MODEL,
            api_key=settings.OPENAI_API_KEY or None,
            dimensions=settings.CACHE_EMBEDDING_DIMENSIONS,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string asynchronously."""
        try:
            vector = await self._embeddings.aembed_query(text)
            return list(vector)
        except Exception:
            logger.exception("openai_embeddings_failed")
            raise
