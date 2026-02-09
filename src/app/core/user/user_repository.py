"""User repository for managing user database operations."""

from typing import Optional

from sqlmodel import Session, select

from src.app.core.common.logging import logger
from src.app.core.user.user_model import User


class UserRepository:
    """Repository class for user database operations.

    This class handles all database operations related to Users.
    """

    def __init__(self, session: Session):
        """Initialize user repository with database session.

        Args:
            session: SQLModel database session
        """
        self.session = session

    async def create_user(self, email: str, password: str) -> User:
        """Create a new user.

        Args:
            email: User's email address
            password: Hashed password

        Returns:
            User: The created user
        """
        user = User(email=email, hashed_password=password)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        logger.info("user_created", email=email)
        return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get a user by ID.

        Args:
            user_id: The ID of the user to retrieve

        Returns:
            Optional[User]: The user if found, None otherwise
        """
        user = self.session.get(User, user_id)
        return user

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email.

        Args:
            email: The email of the user to retrieve

        Returns:
            Optional[User]: The user if found, None otherwise
        """
        statement = select(User).where(User.email == email)
        user = self.session.exec(statement).first()
        return user

    async def delete_user_by_email(self, email: str) -> bool:
        """Delete a user by email.

        Args:
            email: The email of the user to delete

        Returns:
            bool: True if deletion was successful, False if user not found
        """
        user = self.session.exec(select(User).where(User.email == email)).first()
        if not user:
            return False

        self.session.delete(user)
        self.session.commit()
        logger.info("user_deleted", email=email)
        return True