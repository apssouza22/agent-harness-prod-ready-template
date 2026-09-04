"""This file contains the database service for the application."""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import (
    Session,
    SQLModel,
    create_engine,
)

from src.app.core.common.config import (
    Environment,
    Settings,
    settings,
)
from src.app.core.common.logging import logger


class DatabaseFactory:
    """Service class for database operations.

    This class provides access to repositories for User and Session operations.
    It uses SQLModel for ORM operations and maintains a connection pool.
    """

    def __init__(self, app_settings: Settings | None = None):
        """Initialize database service with connection pool."""
        self.settings = app_settings or settings

        try:
            pool_size = self.settings.POSTGRES_POOL_SIZE
            max_overflow = self.settings.POSTGRES_MAX_OVERFLOW

            connection_url = (
                f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
            )

            self.engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,
                pool_recycle=1800,
            )

            SQLModel.metadata.create_all(self.engine)

            logger.info(
                "database_initialized",
                environment=self.settings.ENVIRONMENT.value,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        except SQLAlchemyError as e:
            logger.error(
                "database_initialization_error",
                error=str(e),
                environment=self.settings.ENVIRONMENT.value,
            )
            if self.settings.ENVIRONMENT != Environment.PRODUCTION:
                raise

    def get_session_maker(self) -> Session:
        """Get a session maker for creating database sessions.

        Returns:
            Session: A SQLModel session maker
        """
        return Session(self.engine)

    def dispose(self) -> None:
        """Dispose the database engine and release connection pool resources."""
        self.engine.dispose()
        logger.info("database_disposed")
