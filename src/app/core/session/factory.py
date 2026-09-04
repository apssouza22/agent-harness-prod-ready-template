"""Session repository factory."""

from sqlmodel import Session

from src.app.core.session.session_repository import SessionRepository


def make_session_repository(session: Session) -> SessionRepository:
    """Create a session repository backed by the given database session.

    Args:
        session: SQLModel session for ORM operations.

    Returns:
        SessionRepository: Repository for chat session database operations.
    """
    return SessionRepository(session)
