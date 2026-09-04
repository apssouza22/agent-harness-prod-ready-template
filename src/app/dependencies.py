"""FastAPI dependency getters and typed aliases for app.state singletons."""

from collections.abc import Generator
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langfuse.langchain import CallbackHandler
from sqlmodel import Session

from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.open_deep_research.agent_deep_research import DeepResearchAgent
from src.app.agents.text_to_sql.text_sql_agent import TextSQLDeepAgent
from src.app.api.security.auth import create_access_token, verify_token
from src.app.api.v1.sanitization import sanitize_string
from src.app.core.checkpoint.service import CheckpointService
from src.app.core.common.config import Settings
from src.app.core.common.logging import bind_context, logger
from src.app.core.db.database import DatabaseFactory
from src.app.core.memory.memory import MemoryService
from src.app.core.session.session_model import Session as ChatSession
from src.app.core.session.session_repository import SessionRepository
from src.app.core.user.user_model import User
from src.app.core.user.user_repository import UserRepository

security = HTTPBearer(auto_error=False)

_missing_credentials = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> DatabaseFactory:
    return request.app.state.database


def get_db_session(
    database: Annotated[DatabaseFactory, Depends(get_database)],
) -> Generator[Session, None, None]:
    with Session(database.engine) as session:
        yield session


def get_user_repository(request: Request) -> UserRepository:
    return request.app.state.user_repository


def get_session_repository(request: Request) -> SessionRepository:
    return request.app.state.session_repository


def get_memory_service(request: Request) -> MemoryService:
    return request.app.state.memory_service


def get_checkpoint_service(request: Request) -> CheckpointService:
    return request.app.state.checkpoint_service


def get_langfuse_callback_handler(request: Request) -> CallbackHandler:
    return request.app.state.langfuse_callback_handler


def get_chatbot_agent(request: Request) -> AgentChatbot:
    return request.app.state.chatbot_agent


def get_deep_research_agent(request: Request) -> DeepResearchAgent:
    return request.app.state.deep_research_agent


def get_text_to_sql_agent(request: Request) -> TextSQLDeepAgent:
    return request.app.state.text_to_sql_agent


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[DatabaseFactory, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
SessionRepositoryDep = Annotated[SessionRepository, Depends(get_session_repository)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
CheckpointServiceDep = Annotated[CheckpointService, Depends(get_checkpoint_service)]
LangfuseCallbackHandlerDep = Annotated[CallbackHandler, Depends(get_langfuse_callback_handler)]
ChatbotAgentDep = Annotated[AgentChatbot, Depends(get_chatbot_agent)]
DeepResearchAgentDep = Annotated[DeepResearchAgent, Depends(get_deep_research_agent)]
TextToSqlAgentDep = Annotated[TextSQLDeepAgent, Depends(get_text_to_sql_agent)]


async def get_current_user(
    user_repository: UserRepositoryDep,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    if credentials is None:
        raise _missing_credentials
    try:
        token = sanitize_string(credentials.credentials)
        user_id = verify_token(token)
        if user_id is None:
            logger.error("invalid_token", token_part=token[:10] + "...")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id_int = int(user_id)
        user = await user_repository.get_user(user_id_int)
        if user is None:
            logger.error("user_not_found", user_id=user_id_int)
            raise HTTPException(
                status_code=404,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        bind_context(user_id=user_id_int)
        return user
    except ValueError as ve:
        logger.error("token_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_session(
    session_repository: SessionRepositoryDep,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> ChatSession:
    if credentials is None:
        raise _missing_credentials
    try:
        token = sanitize_string(credentials.credentials)
        session_id = verify_token(token)
        if session_id is None:
            logger.error("session_id_not_found", token_part=token[:10] + "...")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        session_id = sanitize_string(session_id)
        session = await session_repository.get_session(session_id)
        if session is None:
            logger.error("session_not_found", session_id=session_id)
            raise HTTPException(
                status_code=404,
                detail="Session not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        bind_context(user_id=session.user_id)
        return session
    except ValueError as ve:
        logger.error("token_validation_failed", error=str(ve), exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )


CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentSessionDep = Annotated[ChatSession, Depends(get_current_session)]
