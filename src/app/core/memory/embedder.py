"""Embedding providers for long-term memory vector search."""

from langchain_aws import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.app.core.common.config import Settings
from src.app.core.common.logging import logger
from src.app.core.llm.factory import build_bedrock_client_kwargs, build_openai_embeddings_kwargs
from src.app.core.memory.config_builder import (
    normalize_memory_model,
    resolve_memory_provider,
)


class MemoryEmbedder:
    """Generate embeddings for memory storage and retrieval."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._openai_embeddings: OpenAIEmbeddings | None = None
        self._bedrock_embeddings: BedrockEmbeddings | None = None

    def _resolve_provider(self) -> str:
        return resolve_memory_provider(
            self._settings.LONG_TERM_MEMORY_EMBEDDER_PROVIDER,
            self._settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
            fallback_provider=self._settings.LONG_TERM_MEMORY_LLM_PROVIDER or self._settings.DEFAULT_LLM_PROVIDER,
        )

    def _resolve_model(self) -> str:
        _, model = normalize_memory_model(self._settings.LONG_TERM_MEMORY_EMBEDDER_MODEL)
        return model

    def _get_openai_embeddings(self) -> OpenAIEmbeddings:
        if self._openai_embeddings is None:
            kwargs = build_openai_embeddings_kwargs(
                self._settings,
                bifrost_agent="agent_1",
                model=self._resolve_model(),
                dimensions=self._settings.LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
            )
            self._openai_embeddings = OpenAIEmbeddings(**kwargs)
        return self._openai_embeddings

    def _get_bedrock_embeddings(self) -> BedrockEmbeddings:
        if self._bedrock_embeddings is None:
            bedrock_kwargs = build_bedrock_client_kwargs(self._settings)
            self._bedrock_embeddings = BedrockEmbeddings(
                model_id=self._resolve_model(),
                region_name=bedrock_kwargs.get("region_name"),
                credentials_profile_name=bedrock_kwargs.get("credentials_profile_name"),
                aws_access_key_id=bedrock_kwargs.get("aws_access_key_id"),
                aws_secret_access_key=bedrock_kwargs.get("aws_secret_access_key"),
                aws_session_token=bedrock_kwargs.get("aws_session_token"),
            )
        return self._bedrock_embeddings

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string for storage or search."""
        provider = self._resolve_provider()
        try:
            if provider == "bedrock":
                vector = await self._get_bedrock_embeddings().aembed_query(text)
            else:
                vector = await self._get_openai_embeddings().aembed_query(text)
            return list(vector)
        except Exception:
            logger.exception("memory_embedding_failed", provider=provider)
            raise
