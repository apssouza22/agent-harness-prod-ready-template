"""User repository factory."""

from sqlmodel import Session

from src.app.core.user.user_repository import UserRepository


def make_user_repository(session: Session) -> UserRepository:
    """Create a user repository backed by the given database session.

    Args:
        session: SQLModel session for ORM operations.

    Returns:
        UserRepository: Repository for user database operations.
    """
    return UserRepository(session)
