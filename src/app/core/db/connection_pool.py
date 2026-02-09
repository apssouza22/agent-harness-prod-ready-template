
from typing import Optional
from urllib.parse import quote_plus

from psycopg_pool import AsyncConnectionPool

from src.app.core.common.config import Environment, settings
from src.app.core.common.logging import logger

# Module-level singleton for connection pool
_connection_pool: Optional[AsyncConnectionPool] = None


async def get_connection_pool() -> Optional[AsyncConnectionPool]:
  """Get or create PostgreSQL connection pool with environment-specific settings.

  Returns:
      Optional[AsyncConnectionPool]: Connection pool instance, or None if creation fails in production.
  """
  global _connection_pool

  if _connection_pool is None:
    try:
      # Configure pool size based on environment
      max_size = settings.POSTGRES_POOL_SIZE

      connection_url = (
        "postgresql://"
        f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
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
      logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
    except Exception as e:
      logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
      # In production, we might want to degrade gracefully
      if settings.ENVIRONMENT == Environment.PRODUCTION:
        logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
        return None
      raise e
  return _connection_pool

