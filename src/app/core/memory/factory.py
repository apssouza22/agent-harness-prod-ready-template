"""Long-term memory service factory."""

from functools import lru_cache

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.memory.memory import MemoryService


def make_memory_service(app_settings: Settings | None = None) -> MemoryService:
    """Create a memory service for long-term user memory via mem0.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.

    Returns:
        MemoryService: Configured memory service instance.
    """
    resolved_settings = app_settings or default_settings
    return MemoryService(resolved_settings)


@lru_cache(maxsize=1)
def make_memory_service_cached() -> MemoryService:
    """Return a process-wide cached memory service using default settings."""
    return make_memory_service()


def make_memory_service_fresh(app_settings: Settings | None = None) -> MemoryService:
    """Create a new memory service bypassing the process-wide cache."""
    return make_memory_service(app_settings)
