"""Checkpoint and connection pool factories for LangGraph persistence."""

from urllib.parse import quote_plus

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.app.core.common.config import Environment, Settings, settings as default_settings
from src.app.core.common.logging import logger

_connection_pool: AsyncConnectionPool | None = None


async def make_connection_pool(app_settings: Settings | None = None) -> AsyncConnectionPool | None:
    """Create or return the async PostgreSQL connection pool for checkpointing.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.

    Returns:
        AsyncConnectionPool when successful, or None in production on failure.
    """
    global _connection_pool
    resolved_settings = app_settings or default_settings

    if _connection_pool is not None:
        return _connection_pool

    try:
        max_size = resolved_settings.POSTGRES_POOL_SIZE
        connection_url = (
            "postgresql://"
            f"{quote_plus(resolved_settings.POSTGRES_USER)}:{quote_plus(resolved_settings.POSTGRES_PASSWORD)}"
            f"@{resolved_settings.POSTGRES_HOST}:{resolved_settings.POSTGRES_PORT}/{resolved_settings.POSTGRES_DB}"
        )

        _connection_pool = AsyncConnectionPool(
            connection_url,
            open=False,
            max_size=max_size,
            kwargs={
                "autocommit": True,
                "connect_timeout": 5,
                "prepare_threshold": None,
            },
        )
        await _connection_pool.open()
        logger.info(
            "connection_pool_created",
            max_size=max_size,
            environment=resolved_settings.ENVIRONMENT.value,
        )
    except Exception as e:
        logger.error(
            "connection_pool_creation_failed",
            error=str(e),
            environment=resolved_settings.ENVIRONMENT.value,
        )
        if resolved_settings.ENVIRONMENT == Environment.PRODUCTION:
            logger.warning(
                "continuing_without_connection_pool",
                environment=resolved_settings.ENVIRONMENT.value,
            )
            return None
        raise

    return _connection_pool


async def make_checkpointer(
    connection_pool: AsyncConnectionPool | None = None,
    app_settings: Settings | None = None,
) -> AsyncPostgresSaver | None:
    """Create a LangGraph async Postgres checkpointer.

    Args:
        connection_pool: Optional pre-built pool. Created via make_connection_pool when omitted.
        app_settings: Application settings. Falls back to the module-level singleton.

    Returns:
        AsyncPostgresSaver when a pool is available, otherwise None in production.
    """
    resolved_settings = app_settings or default_settings
    pool = connection_pool if connection_pool is not None else await make_connection_pool(resolved_settings)

    if pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        return checkpointer

    if resolved_settings.ENVIRONMENT != Environment.PRODUCTION:
        raise RuntimeError("connection pool initialization failed")

    return None


async def reset_connection_pool() -> None:
    """Close and clear the cached connection pool. Intended for tests."""
    global _connection_pool

    if _connection_pool is None:
        return

    await _connection_pool.close()
    _connection_pool = None
