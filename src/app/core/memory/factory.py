"""Long-term memory service factory."""

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.memory.memory import MemoryService


def make_memory_service(app_settings: Settings | None = None) -> MemoryService:
    """Create a memory service for long-term user memory via pgvector.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.

    Returns:
        MemoryService: Configured memory service instance.
    """
    resolved_settings = app_settings or default_settings
    return MemoryService(resolved_settings)


def make_memory_service_fresh(app_settings: Settings | None = None) -> MemoryService:
    """Create a new memory service instance.

    Useful in tests that need an isolated service.
    """
    return make_memory_service(app_settings)
