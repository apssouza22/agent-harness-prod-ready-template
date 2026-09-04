"""Checkpoint service factory for LangGraph persistence."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.app.core.checkpoint.service import CheckpointService
from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.db.connection_pool import get_connection_pool


def make_checkpoint_service(
    app_settings: Settings | None = None,
    *,
    connection_pool: AsyncConnectionPool | None,
) -> CheckpointService:
    """Create a checkpoint service for LangGraph persistence.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.
        connection_pool: Async PostgreSQL pool injected by the caller.

    Returns:
        CheckpointService: Configured checkpoint service instance.
    """
    resolved_settings = app_settings or default_settings
    return CheckpointService(resolved_settings, connection_pool)


async def make_checkpointer(
    app_settings: Settings | None = None,
    *,
    connection_pool: AsyncConnectionPool | None = None,
) -> AsyncPostgresSaver | None:
    """Create a LangGraph async Postgres checkpointer.

    Backward-compatible wrapper around CheckpointService.get_checkpointer().
    """
    resolved_settings = app_settings or default_settings
    pool = connection_pool if connection_pool is not None else await get_connection_pool(resolved_settings)
    service = make_checkpoint_service(resolved_settings, connection_pool=pool)
    return await service.get_checkpointer()
