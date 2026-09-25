"""Long-term memory chat model construction via the shared LLM factory."""

from langchain_core.language_models.chat_models import BaseChatModel

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.llm.factory import make_chat_model, resolve_model_identifier
from src.app.core.memory.config_builder import resolve_memory_provider


def make_memory_chat_model(app_settings: Settings | None = None) -> BaseChatModel:
    """Create the chat model used for memory extraction, reconciliation, and entity parsing."""
    resolved_settings = app_settings or default_settings
    provider = resolve_memory_provider(
        resolved_settings.LONG_TERM_MEMORY_LLM_PROVIDER,
        resolved_settings.LONG_TERM_MEMORY_MODEL,
        fallback_provider=resolved_settings.DEFAULT_LLM_PROVIDER,
    )
    model_name = resolve_model_identifier(
        resolved_settings.LONG_TERM_MEMORY_MODEL,
        provider,
        resolved_settings,
    )
    return make_chat_model(
        model_name,
        app_settings=resolved_settings,
        bifrost_agent="agent_1",
        max_tokens=resolved_settings.MAX_TOKENS,
    )
