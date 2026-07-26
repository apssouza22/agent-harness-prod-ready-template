"""Unit tests for Bifrost LLM factory helpers."""

import pytest

from src.app.core.common import config as config_module
from src.app.core.llm import factory


@pytest.fixture
def bifrost_settings(monkeypatch):
    """Enable Bifrost routing for factory tests."""
    monkeypatch.setenv("BIFROST_ENABLED", "true")
    monkeypatch.setenv("BIFROST_BASE_URL", "http://bifrost:8080/langchain")
    monkeypatch.setenv("BIFROST_OPENAI_BASE_URL", "http://bifrost:8080/v1")
    monkeypatch.setenv("BIFROST_API_KEY", "test-dummy-key")
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "test-virtual-key")
    updated_settings = config_module.Settings()
    monkeypatch.setattr(factory, "settings", updated_settings)
    return updated_settings


@pytest.fixture
def direct_settings(monkeypatch):
    """Disable Bifrost routing for factory tests."""
    monkeypatch.setenv("BIFROST_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    updated_settings = config_module.Settings()
    monkeypatch.setattr(factory, "settings", updated_settings)
    return updated_settings


def test_build_chat_model_kwargs_with_bifrost(bifrost_settings):
    kwargs = factory.build_chat_model_kwargs(model="openai:gpt-4o-mini", max_tokens=100)

    assert kwargs["base_url"] == "http://bifrost:8080/langchain"
    assert kwargs["api_key"] == "test-dummy-key"
    assert kwargs["default_headers"] == {"x-bf-vk": "test-virtual-key"}
    assert kwargs["max_tokens"] == 100


def test_build_chat_model_kwargs_without_bifrost(direct_settings):
    kwargs = factory.build_chat_model_kwargs(model="openai:gpt-4o-mini", max_tokens=100)

    assert "base_url" not in kwargs
    assert kwargs["api_key"] == "sk-test-openai"


def test_build_openai_client_kwargs_with_bifrost(bifrost_settings):
    kwargs = factory.build_openai_client_kwargs()

    assert kwargs["api_key"] == "test-dummy-key"
    assert kwargs["base_url"] == "http://bifrost:8080/v1"


def test_build_mem0_openai_config_with_bifrost(bifrost_settings):
    mem0_config = factory.build_mem0_openai_config()

    assert mem0_config["api_key"] == "test-dummy-key"
    assert mem0_config["openai_base_url"] == "http://bifrost:8080/v1"


def test_resolve_api_key_for_model_uses_bifrost_key(bifrost_settings):
    assert factory.resolve_api_key_for_model("openai:gpt-4o-mini") == "test-dummy-key"


def test_create_chat_model_uses_bifrost_base_url(bifrost_settings):
    model = factory.create_chat_model("openai:gpt-4o-mini", max_tokens=50)

    assert str(model.root_client.base_url).rstrip("/") == "http://bifrost:8080/langchain"
