"""Long-term memory service factory."""

from langchain_core.language_models.chat_models import BaseChatModel

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.memory.llm import make_memory_chat_model
from src.app.core.memory.memory import MemoryService


def make_memory_service(
    app_settings: Settings | None = None,
    chat_model: BaseChatModel | None = None,
) -> MemoryService:
    """Create a memory service for long-term user memory via pgvector.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.
        chat_model: Optional pre-built chat model for tests or custom wiring.

    Returns:
        MemoryService: Configured memory service instance.
    """
    resolved_settings = app_settings or default_settings
    llm = chat_model or make_memory_chat_model(resolved_settings)
    return MemoryService(resolved_settings, chat_model=llm)


def make_memory_service_fresh(
    app_settings: Settings | None = None,
    chat_model: BaseChatModel | None = None,
) -> MemoryService:
    """Create a new memory service instance.

    Useful in tests that need an isolated service.
    """
    return make_memory_service(app_settings, chat_model=chat_model)
