"""Authentication and authorization endpoints for the API.

This module provides endpoints for user registration, login, session management,
and token verification.
"""

import uuid
from typing import List

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
)

from src.app.api.security.auth import create_access_token
from src.app.api.security.limiter import limiter
from src.app.api.v1.sanitization import (
    sanitize_email,
    sanitize_string,
    validate_password_strength,
)
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.common.token_dtos import TokenResponse
from src.app.core.session.session_dto import SessionResponse
from src.app.core.user.user_dtos import UserCreate, UserResponse
from src.app.core.user.user_model import User
from src.app.dependencies import (
    CurrentSessionDep,
    CurrentUserDep,
    SessionRepositoryDep,
    UserRepositoryDep,
)

router = APIRouter()


@router.post("/register", response_model=UserResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["register"][0])
async def register_user(
    request: Request,
    user_data: UserCreate,
    user_repository: UserRepositoryDep,
):
    """Register a new user."""
    try:
        sanitized_email = sanitize_email(user_data.email)
        password = user_data.password.get_secret_value()
        validate_password_strength(password)

        if await user_repository.get_user_by_email(sanitized_email):
            raise HTTPException(status_code=400, detail="Email already registered")

        user = await user_repository.create_user(email=sanitized_email, password=User.hash_password(password))
        token = create_access_token(str(user.id))
        return UserResponse(id=user.id, email=user.email, token=token)
    except ValueError as ve:
        logger.error("user_registration_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["login"][0])
async def login(
    request: Request,
    user_repository: UserRepositoryDep,
    username: str = Form(...),
    password: str = Form(...),
    grant_type: str = Form(default="password"),
):
    """Login a user."""
    try:
        username = sanitize_string(username)
        password = sanitize_string(password)
        grant_type = sanitize_string(grant_type)

        if grant_type != "password":
            raise HTTPException(
                status_code=400,
                detail="Unsupported grant type. Must be 'password'",
            )

        user = await user_repository.get_user_by_email(username)
        if not user or not user.verify_password(password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = create_access_token(str(user.id))
        return TokenResponse(access_token=token.access_token, token_type="bearer", expires_at=token.expires_at)
    except ValueError as ve:
        logger.error("login_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))


@router.post("/session", response_model=SessionResponse)
async def create_session(
    user: CurrentUserDep,
    session_repository: SessionRepositoryDep,
):
    """Create a new chat session for the authenticated user."""
    try:
        session_id = str(uuid.uuid4())
        session = await session_repository.create_session(session_id, user.id)
        token = create_access_token(session_id)

        logger.info(
            "session_created",
            session_id=session_id,
            user_id=user.id,
            name=session.name,
            expires_at=token.expires_at.isoformat(),
        )

        return SessionResponse(session_id=session_id, name=session.name, token=token)
    except ValueError as ve:
        logger.error("session_creation_validation_failed", error=str(ve), user_id=user.id, exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))


@router.patch("/session/{session_id}/name", response_model=SessionResponse)
async def update_session_name(
    session_id: str,
    current_session: CurrentSessionDep,
    session_repository: SessionRepositoryDep,
    name: str = Form(...),
):
    """Update a session's name."""
    try:
        sanitized_session_id = sanitize_string(session_id)
        sanitized_name = sanitize_string(name)
        sanitized_current_session = sanitize_string(current_session.id)

        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot modify other sessions")

        session = await session_repository.update_session_name(sanitized_session_id, sanitized_name)
        token = create_access_token(sanitized_session_id)
        return SessionResponse(session_id=sanitized_session_id, name=session.name, token=token)
    except ValueError as ve:
        logger.error("session_update_validation_failed", error=str(ve), session_id=session_id, exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    current_session: CurrentSessionDep,
    session_repository: SessionRepositoryDep,
):
    """Delete a session for the authenticated user."""
    try:
        sanitized_session_id = sanitize_string(session_id)
        sanitized_current_session = sanitize_string(current_session.id)

        if sanitized_session_id != sanitized_current_session:
            raise HTTPException(status_code=403, detail="Cannot delete other sessions")

        await session_repository.delete_session(sanitized_session_id)
        logger.info("session_deleted", session_id=session_id, user_id=current_session.user_id)
    except ValueError as ve:
        logger.error("session_deletion_validation_failed", error=str(ve), session_id=session_id, exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))


@router.get("/sessions", response_model=List[SessionResponse])
async def get_user_sessions(
    user: CurrentUserDep,
    session_repository: SessionRepositoryDep,
):
    """Get all session IDs for the authenticated user."""
    try:
        sessions = await session_repository.get_user_sessions(user.id)
        return [
            SessionResponse(
                session_id=sanitize_string(session.id),
                name=sanitize_string(session.name),
                token=create_access_token(session.id),
            )
            for session in sessions
        ]
    except ValueError as ve:
        logger.error("get_sessions_validation_failed", user_id=user.id, error=str(ve), exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))
