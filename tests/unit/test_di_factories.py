"""Unit tests for make_* dependency injection factories."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session

from src.app.core.checkpoint.factory import make_checkpointer, make_connection_pool, reset_connection_pool
from src.app.core.common import config as config_module
from src.app.core.db.factory import make_database, make_database_fresh
from src.app.core.llm import factory as llm_factory
from src.app.core.memory.factory import make_memory_service, make_memory_service_fresh
from src.app.core.session.factory import make_session_repository
from src.app.core.tracing.factory import (
    init_langfuse,
    make_langfuse_callback_handler,
    shutdown_langfuse,
)
from src.app.core.user.factory import make_user_repository


@pytest.fixture
def test_settings(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return config_module.Settings()


def test_make_database_returns_factory_with_engine(test_settings):
    with patch("src.app.core.db.database.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        database = make_database_fresh(test_settings)

    assert database.engine is mock_engine
    assert database.settings is test_settings


def test_make_user_repository_returns_repository():
    session = MagicMock(spec=Session)
    repository = make_user_repository(session)

    assert repository.session is session


def test_make_session_repository_returns_repository():
    session = MagicMock(spec=Session)
    repository = make_session_repository(session)

    assert repository.session is session


def test_make_memory_service_accepts_settings(test_settings):
    service = make_memory_service(test_settings)

    assert service._settings is test_settings


def test_make_memory_service_fresh_bypasses_cache(test_settings):
    first = make_memory_service_fresh(test_settings)
    second = make_memory_service_fresh(test_settings)

    assert first is not second


def test_make_chat_model_uses_injected_settings(monkeypatch, test_settings):
    monkeypatch.setenv("BIFROST_ENABLED", "true")
    monkeypatch.setenv("BIFROST_BASE_URL", "http://bifrost:8080/langchain")
    monkeypatch.setenv("BIFROST_API_KEY", "test-dummy-key")
    bifrost_settings = config_module.Settings()

    kwargs = llm_factory.build_chat_model_kwargs(app_settings=bifrost_settings, model="openai:gpt-4o-mini")

    assert kwargs["base_url"] == "http://bifrost:8080/langchain"
    assert kwargs["api_key"] == "test-dummy-key"


def test_make_langfuse_callback_handler_returns_handler():
    handler = make_langfuse_callback_handler()

    assert handler is not None


def test_init_and_shutdown_langfuse(test_settings):
    mock_client = MagicMock()
    mock_client.auth_check.return_value = True

    with patch("src.app.core.tracing.factory.get_client", return_value=mock_client):
        init_langfuse(test_settings)
        shutdown_langfuse()

    mock_client.auth_check.assert_called_once()
    mock_client.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_make_checkpointer_uses_injected_pool(test_settings):
    mock_pool = AsyncMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.setup = AsyncMock()

    with patch(
        "src.app.core.checkpoint.factory.AsyncPostgresSaver",
        return_value=mock_checkpointer,
    ) as mock_saver_cls:
        checkpointer = await make_checkpointer(connection_pool=mock_pool, app_settings=test_settings)

    mock_saver_cls.assert_called_once_with(mock_pool)
    mock_checkpointer.setup.assert_awaited_once()
    assert checkpointer is mock_checkpointer


@pytest.mark.asyncio
async def test_make_connection_pool_can_be_reset(test_settings):
    with patch("src.app.core.checkpoint.factory.AsyncConnectionPool") as mock_pool_cls:
        mock_pool = AsyncMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_pool_cls.return_value = mock_pool

        pool = await make_connection_pool(test_settings)
        assert pool is mock_pool

        await reset_connection_pool()
        mock_pool.close.assert_awaited_once()

        pool_again = await make_connection_pool(test_settings)
        assert pool_again is mock_pool
