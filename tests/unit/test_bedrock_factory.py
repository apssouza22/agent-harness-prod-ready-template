"""Unit tests for AWS Bedrock LLM factory helpers."""

import pytest

from src.app.core.common import config as config_module
from src.app.core.llm import factory


@pytest.fixture
def bedrock_settings(monkeypatch):
    """Enable direct Bedrock routing for factory tests."""
    monkeypatch.setenv("BIFROST_ENABLED", "false")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test-session-token")
    updated_settings = config_module.Settings()
    monkeypatch.setattr(factory, "settings", updated_settings)
    return updated_settings


@pytest.fixture
def openai_settings(monkeypatch):
    """Default OpenAI provider settings for factory tests."""
    monkeypatch.setenv("BIFROST_ENABLED", "false")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    updated_settings = config_module.Settings()
    monkeypatch.setattr(factory, "settings", updated_settings)
    return updated_settings


def test_resolve_model_identifier_adds_provider_prefix(openai_settings):
    assert factory.resolve_model_identifier() == "openai:gpt-5.6-luna"
    assert factory.resolve_model_identifier("gpt-4o-mini") == "openai:gpt-4o-mini"
    assert factory.resolve_model_identifier("gpt-4o-mini", provider="bedrock") == "bedrock:gpt-4o-mini"


def test_resolve_model_identifier_passthrough_prefixed_model(openai_settings):
    model_id = "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    assert factory.resolve_model_identifier(model_id) == model_id


def test_build_bedrock_client_kwargs(bedrock_settings):
    kwargs = factory.build_bedrock_client_kwargs()

    assert kwargs == {
        "region_name": "us-west-2",
        "credentials_profile_name": "test-profile",
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_session_token": "test-session-token",
    }


def test_build_chat_model_kwargs_for_bedrock(bedrock_settings):
    model_id = "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    kwargs = factory.build_chat_model_kwargs(
        model=model_id,
        max_tokens=100,
        reasoning={"effort": "low"},
        temperature=0.2,
    )

    assert kwargs["model"] == model_id
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.2
    assert kwargs["region_name"] == "us-west-2"
    assert kwargs["credentials_profile_name"] == "test-profile"
    assert "api_key" not in kwargs
    assert "reasoning" not in kwargs


def test_build_chat_model_kwargs_for_openai_keeps_reasoning(openai_settings):
    kwargs = factory.build_chat_model_kwargs(
        model="openai:gpt-5.6-luna",
        reasoning={"effort": "low"},
    )

    assert kwargs["api_key"] == "sk-test-openai"
    assert kwargs["reasoning"] == {"effort": "low"}


def test_resolve_api_key_for_model_returns_empty_for_bedrock(bedrock_settings):
    assert factory.resolve_api_key_for_model("bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0") == ""


def test_bifrost_takes_precedence_over_bedrock(monkeypatch, bedrock_settings):
    monkeypatch.setenv("BIFROST_ENABLED", "true")
    monkeypatch.setenv("BIFROST_BASE_URL", "http://bifrost:8080/langchain")
    monkeypatch.setenv("BIFROST_API_KEY", "test-dummy-key")
    bifrost_settings = config_module.Settings()
    monkeypatch.setattr(factory, "settings", bifrost_settings)

    kwargs = factory.build_chat_model_kwargs(
        model="bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        reasoning={"effort": "low"},
    )

    assert kwargs["base_url"] == "http://bifrost:8080/langchain"
    assert kwargs["api_key"] == "test-dummy-key"
    assert "region_name" not in kwargs
    assert "reasoning" not in kwargs
