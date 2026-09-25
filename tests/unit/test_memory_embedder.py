"""Unit tests for memory embedder Bifrost routing."""

import pytest

from src.app.core.common import config as config_module
from src.app.core.llm import factory
from src.app.core.memory.embedder import MemoryEmbedder


@pytest.fixture
def bifrost_settings(monkeypatch):
    monkeypatch.setenv("BIFROST_ENABLED", "true")
    monkeypatch.setenv("BIFROST_BASE_URL", "http://localhost:8090/langchain")
    monkeypatch.setenv("BIFROST_API_KEY", "test-dummy-key")
    monkeypatch.setenv("BIFROST_API_KEY_AGENT_1", "sk-bf-agent-1-test")
    return config_module.Settings()


@pytest.fixture
def direct_settings(monkeypatch):
    monkeypatch.setenv("BIFROST_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    return config_module.Settings()


def test_derive_bifrost_openai_base_url_from_langchain_url():
    derived = config_module.Settings._derive_bifrost_openai_base_url("http://localhost:8090/langchain")
    assert derived == "http://localhost:8090/v1"


def test_settings_default_bifrost_openai_base_url(bifrost_settings):
    assert bifrost_settings.BIFROST_OPENAI_BASE_URL == "http://localhost:8090/v1"


def test_build_openai_embeddings_kwargs_with_bifrost(bifrost_settings):
    kwargs = factory.build_openai_embeddings_kwargs(
        bifrost_settings,
        bifrost_agent="agent_1",
        model="text-embedding-3-small",
        dimensions=1536,
    )

    assert kwargs["base_url"] == "http://localhost:8090/v1"
    assert kwargs["api_key"] == "test-dummy-key"
    assert kwargs["default_headers"] == {"x-bf-vk": "sk-bf-agent-1-test"}
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["dimensions"] == 1536


def test_build_openai_embeddings_kwargs_without_bifrost(direct_settings):
    kwargs = factory.build_openai_embeddings_kwargs(
        direct_settings,
        model="text-embedding-3-small",
        dimensions=1536,
    )

    assert "base_url" not in kwargs
    assert kwargs["api_key"] == "sk-test-openai"


def test_memory_embedder_uses_bifrost_openai_base_url(bifrost_settings):
    embedder = MemoryEmbedder(bifrost_settings)
    embeddings = embedder._get_openai_embeddings()

    assert str(embeddings.async_client._client.base_url).rstrip("/") == "http://localhost:8090/v1"
    assert embeddings.async_client._client.api_key == "test-dummy-key"
    assert embeddings.async_client._client.default_headers.get("x-bf-vk") == "sk-bf-agent-1-test"
