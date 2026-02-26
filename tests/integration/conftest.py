"""Shared fixtures for integration tests.

Patches the database engine, Langfuse, and all agent factories so that tests
run against an in-memory SQLite database with no real OpenAI or external calls.

IMPORTANT: environment variables and monkey-patches at the top of this module
run *before* any application code is imported.
"""

import os

# ---------------------------------------------------------------------------
# Environment must be set BEFORE any application module is imported
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Intercept create_engine so DatabaseFactory uses an in-memory SQLite DB
# ---------------------------------------------------------------------------

_original_create_engine = _sqlmodel_module.create_engine
_shared_engine = None


def _sqlite_create_engine(*args, **kwargs):
    """Replace any engine creation with a shared in-memory SQLite engine."""
    global _shared_engine
    if _shared_engine is None:
        _shared_engine = _original_create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
    return _shared_engine


_sqlmodel_module.create_engine = _sqlite_create_engine

# Prevent Langfuse from making real network calls during module-level init
_mock_langfuse_inst = MagicMock()
_mock_langfuse_inst.auth_check.return_value = True
patch("langfuse.Langfuse", return_value=_mock_langfuse_inst).start()
patch("langfuse.langchain.CallbackHandler", return_value=MagicMock()).start()

# ---------------------------------------------------------------------------
# Now safe to import application code.
# Importing the app triggers DatabaseFactory() which calls our patched
# create_engine and sets _shared_engine.
# ---------------------------------------------------------------------------

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import Session, SQLModel

from src.app.core.common.model.message import Message
from src.app.main import app as _app  # triggers engine + table creation

TEST_PASSWORD = "TestPass123!"
TEST_EMAIL = "testuser@example.com"


# ---------------------------------------------------------------------------
# Mock agent helpers (no real OpenAI calls)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Application & client fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Build a fully-patched ASGI test client.

    For every test a fresh set of DB tables, repositories, and mock agents is
    created so tests remain isolated from each other.
    """
    SQLModel.metadata.drop_all(_shared_engine)
    SQLModel.metadata.create_all(_shared_engine)

    db_session = Session(_shared_engine)

    from src.app.core.session import SessionRepository
    from src.app.core.user import UserRepository

    test_user_repo = UserRepository(db_session)
    test_session_repo = SessionRepository(db_session)

    # auth.py calls session-related methods on user_repository; bridge them
    test_user_repo.update_session_name = test_session_repo.update_session_name
    test_user_repo.delete_session = test_session_repo.delete_session
    test_user_repo.get_user_sessions = test_session_repo.get_user_sessions

    from src.app.api.security.limiter import limiter

    limiter.reset()

    with (
        patch("src.app.api.v1.api.user_repository", test_user_repo),
        patch("src.app.api.v1.auth.user_repository", test_user_repo),
        patch("src.app.api.v1.auth.session_repository", test_session_repo),
        patch(
            "src.app.api.v1.chatbot.get_agent_example",
            new_callable=AsyncMock,
            return_value=_make_mock_chatbot_agent(),
        ),
        patch(
            "src.app.api.v1.deep_research.get_deep_research_agent",
            new_callable=AsyncMock,
            return_value=_make_mock_deep_research_agent(),
        ),
        patch(
            "src.app.api.v1.text_to_sql.get_text_sql_agent",
            new_callable=AsyncMock,
            return_value=_make_mock_text_sql_agent(),
        ),
        patch("src.app.api.v1.chatbot.clear_checkpoints", new_callable=AsyncMock),
    ):
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac

    db_session.close()


# ---------------------------------------------------------------------------
# Auth helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def registered_user(client: AsyncClient) -> dict:
    """Register a test user and return the response payload."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture()
async def user_token(registered_user: dict) -> str:
    """Return the bearer token for the registered test user."""
    return registered_user["token"]["access_token"]


@pytest.fixture()
async def session_with_token(client: AsyncClient, user_token: str) -> dict:
    """Create a chat session and return its response payload."""
    response = await client.post(
        "/api/v1/auth/session",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture()
def session_token(session_with_token: dict) -> str:
    """Return the bearer token scoped to a chat session."""
    return session_with_token["token"]["access_token"]


@pytest.fixture()
def auth_headers(session_token: str) -> dict:
    """Return Authorization headers for a chat-session-scoped token."""
    return {"Authorization": f"Bearer {session_token}"}
