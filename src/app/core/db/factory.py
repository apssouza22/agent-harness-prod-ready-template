"""Database factory for constructing database engines and session makers."""

from functools import lru_cache

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.db.database import DatabaseFactory


def make_database(app_settings: Settings | None = None) -> DatabaseFactory:
    """Create a database factory with connection pool and ORM engine.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.

    Returns:
        DatabaseFactory: Configured database factory instance.
    """
    resolved_settings = app_settings or default_settings
    return DatabaseFactory(resolved_settings)


def make_database_fresh(app_settings: Settings | None = None) -> DatabaseFactory:
    """Create a new database factory bypassing the process-wide cache.

    Useful in tests that need an isolated engine.
    """
    return make_database(app_settings)


@lru_cache(maxsize=1)
def make_database_cached() -> DatabaseFactory:
    """Return a process-wide cached database factory using default settings."""
    return make_database()
