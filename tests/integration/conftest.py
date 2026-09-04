"""Shared fixtures for integration tests.

Patches the database engine, Langfuse, and agent dependencies so that tests
run against an in-memory SQLite database with no real OpenAI or external calls.

IMPORTANT: environment variables and monkey-patches at the top of this module
run *before* any application code is imported.
"""

import os

os.environ["APP_ENV"] = "test"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-integration-tests"
os.environ["OPENAI_API_KEY"] = "sk-test-fake-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-test"
os.environ["LANGFUSE_HOST"] = "http://localhost:0"
os.environ["MCP_ENABLED"] = "false"

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import sqlmodel as _sqlmodel_module

_original_create_engine = _sqlmodel_module.create_engine
_shared_engine = None


def _sqlite_create_engine(*args, **kwargs):
    global _shared_engine
    if _shared_engine is None:
        _shared_engine = _original_create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
    return _shared_engine


_sqlmodel_module.create_engine = _sqlite_create_engine

# Ensure the shared in-memory engine exists before any app lifespan runs.
_sqlite_create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

_mock_langfuse_inst = MagicMock()
_mock_langfuse_inst.auth_check.return_value = True
patch("langfuse.Langfuse", return_value=_mock_langfuse_inst).start()
patch("langfuse.get_client", return_value=_mock_langfuse_inst).start()
patch("langfuse.langchain.CallbackHandler", return_value=MagicMock()).start()

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import Session, SQLModel

from src.app import dependencies
from src.app.core.common.model.message import Message
from src.app.core.db.factory import make_database_cached
from src.app.main import app as _app

TEST_PASSWORD = "TestPass123!"
TEST_EMAIL = "testuser@example.com"


def _make_mock_chatbot_agent():
    agent = AsyncMock()
    agent.name = "Agent Example"
    agent.agent_invoke = AsyncMock(
        return_value=[Message(role="assistant", content="Hello! How can I help you?")]
    )

    async def _fake_stream(*_args, **_kwargs):
        for chunk in ["Hello", " from", " stream"]:
            yield chunk

    agent.agent_invoke_stream = _fake_stream
    agent.get_chat_history = AsyncMock(
        return_value=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
    )
    return agent


def _make_mock_deep_research_agent():
    agent = AsyncMock()
    agent.name = "Deep Research"
    agent.agent_invoke = AsyncMock(
        return_value=[Message(role="assistant", content="Here is your research report.")]
    )

    async def _fake_stream(*_args, **_kwargs):
        for chunk in ["Research", " report", " streaming"]:
            yield chunk

    agent.agent_invoke_stream = _fake_stream
    return agent


def _make_mock_text_sql_agent():
    agent = AsyncMock()
    agent.name = "Text-to-SQL"
    agent.agent_invoke = AsyncMock(
        return_value=[Message(role="assistant", content="SELECT * FROM users;")]
    )
    return agent


@pytest.fixture()
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Build a fully-patched ASGI test client with dependency overrides."""
    make_database_cached.cache_clear()

    SQLModel.metadata.drop_all(_shared_engine)
    SQLModel.metadata.create_all(_shared_engine)

    db_session = Session(_shared_engine)

    from src.app.core.session import SessionRepository
    from src.app.core.user import UserRepository

    test_user_repo = UserRepository(db_session)
    test_session_repo = SessionRepository(db_session)

    from src.app.api.security.limiter import limiter

    limiter.reset()

    _app.dependency_overrides[dependencies.get_user_repository] = lambda: test_user_repo
    _app.dependency_overrides[dependencies.get_session_repository] = lambda: test_session_repo
    mock_checkpoint_service = AsyncMock()
    mock_checkpoint_service.get_checkpointer = AsyncMock(return_value=None)
    mock_checkpoint_service.clear_session = AsyncMock()

    _app.dependency_overrides[dependencies.get_checkpoint_service] = lambda: mock_checkpoint_service
    _app.dependency_overrides[dependencies.get_chatbot_agent] = lambda: _make_mock_chatbot_agent()
    _app.dependency_overrides[dependencies.get_deep_research_agent] = lambda: _make_mock_deep_research_agent()
    _app.dependency_overrides[dependencies.get_text_to_sql_agent] = lambda: _make_mock_text_sql_agent()

    with (
        patch("src.app.main.make_checkpoint_service", return_value=mock_checkpoint_service),
        patch(
            "src.app.main.make_chatbot_agent",
            new_callable=AsyncMock,
            return_value=_make_mock_chatbot_agent(),
        ),
        patch(
            "src.app.main.make_deep_research_agent",
            new_callable=AsyncMock,
            return_value=_make_mock_deep_research_agent(),
        ),
        patch(
            "src.app.main.make_text_to_sql_agent",
            new_callable=AsyncMock,
            return_value=_make_mock_text_sql_agent(),
        ),
    ):
        async with _app.router.lifespan_context(_app):
            transport = ASGITransport(app=_app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                yield ac
            db_session.close()

    _app.dependency_overrides.clear()


@pytest.fixture()
async def registered_user(client: AsyncClient) -> dict:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture()
async def user_token(registered_user: dict) -> str:
    return registered_user["token"]["access_token"]


@pytest.fixture()
async def session_with_token(client: AsyncClient, user_token: str) -> dict:
    response = await client.post(
        "/api/v1/auth/session",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture()
def session_token(session_with_token: dict) -> str:
    return session_with_token["token"]["access_token"]


@pytest.fixture()
def auth_headers(session_token: str) -> dict:
    return {"Authorization": f"Bearer {session_token}"}
