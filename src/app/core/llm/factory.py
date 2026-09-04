"""Centralized LLM factory with optional Bifrost API gateway routing.

When BIFROST_ENABLED is true, LangChain chat models are pointed at the Bifrost
/langchain proxy and OpenAI-compatible clients use the /v1 endpoint. Provider
API keys are managed by Bifrost instead of the application.
"""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.common.logging import logger


settings = default_settings


def _resolve_settings(app_settings: Settings | None = None) -> Settings:
    return app_settings or settings


def get_bifrost_langchain_base_url(app_settings: Settings | None = None) -> str:
    """Return the Bifrost LangChain proxy base URL."""
    resolved_settings = _resolve_settings(app_settings)
    return resolved_settings.BIFROST_BASE_URL.rstrip("/")


def get_bifrost_openai_base_url(app_settings: Settings | None = None) -> str:
    """Return the Bifrost OpenAI-compatible base URL for direct OpenAI clients."""
    resolved_settings = _resolve_settings(app_settings)
    return resolved_settings.BIFROST_OPENAI_BASE_URL.rstrip("/")


def get_bifrost_default_headers(app_settings: Settings | None = None) -> dict[str, str]:
    """Return optional Bifrost governance headers (e.g. virtual key)."""
    resolved_settings = _resolve_settings(app_settings)
    if not resolved_settings.BIFROST_VIRTUAL_KEY:
        return {}
    return {"x-bf-vk": resolved_settings.BIFROST_VIRTUAL_KEY}


def resolve_api_key_for_model(model_name: str | None = None, app_settings: Settings | None = None) -> str:
    """Resolve the API key to pass to LangChain for the given model."""
    resolved_settings = _resolve_settings(app_settings)
    if resolved_settings.BIFROST_ENABLED:
        return resolved_settings.BIFROST_API_KEY

    if model_name:
        normalized = model_name.lower()
        if normalized.startswith("openai:"):
            return resolved_settings.OPENAI_API_KEY
        if normalized.startswith("anthropic:"):
            return resolved_settings.ANTHROPIC_API_KEY or ""
        if normalized.startswith("google"):
            return resolved_settings.GOOGLE_API_KEY or ""

    return resolved_settings.OPENAI_API_KEY


def build_chat_model_kwargs(app_settings: Settings | None = None, **overrides: Any) -> dict[str, Any]:
    """Build kwargs for LangChain chat model initialization."""
    resolved_settings = _resolve_settings(app_settings)
    kwargs: dict[str, Any] = dict(overrides)

    if resolved_settings.BIFROST_ENABLED:
        kwargs["base_url"] = get_bifrost_langchain_base_url(resolved_settings)
        kwargs["api_key"] = resolved_settings.BIFROST_API_KEY
        bifrost_headers = get_bifrost_default_headers(resolved_settings)
        if bifrost_headers:
            existing_headers = kwargs.get("default_headers", {})
            kwargs["default_headers"] = {**existing_headers, **bifrost_headers}
        logger.debug(
            "bifrost_chat_model_routing_enabled",
            base_url=kwargs["base_url"],
            has_virtual_key=bool(resolved_settings.BIFROST_VIRTUAL_KEY),
        )
        return kwargs

    if "api_key" not in kwargs and "model" in kwargs:
        kwargs["api_key"] = resolve_api_key_for_model(kwargs["model"], resolved_settings)

    return kwargs


def create_chat_model(
    model: str,
    *,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    app_settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain chat model, routing through Bifrost when enabled."""
    model_kwargs = build_chat_model_kwargs(app_settings, model=model, **kwargs)
    if configurable_fields is not None:
        return init_chat_model(configurable_fields=configurable_fields, **model_kwargs)
    return init_chat_model(**model_kwargs)


def create_configurable_chat_model(
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ("model", "max_tokens", "api_key"),
    app_settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a configurable chat model with Bifrost defaults applied."""
    model_kwargs = build_chat_model_kwargs(app_settings, **kwargs)
    return init_chat_model(configurable_fields=configurable_fields, **model_kwargs)


def create_openai_chat_model(app_settings: Settings | None = None, **kwargs: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance, routing through Bifrost when enabled."""
    model_kwargs = build_chat_model_kwargs(app_settings, **kwargs)
    return ChatOpenAI(**model_kwargs)


def build_openai_client_kwargs(app_settings: Settings | None = None, **overrides: Any) -> dict[str, Any]:
    """Build kwargs for OpenAI SDK clients (mem0, evaluations, etc.)."""
    resolved_settings = _resolve_settings(app_settings)
    if resolved_settings.BIFROST_ENABLED:
        return {
            "api_key": resolved_settings.BIFROST_API_KEY,
            "base_url": get_bifrost_openai_base_url(resolved_settings),
            **overrides,
        }

    api_key = overrides.pop("api_key", resolved_settings.OPENAI_API_KEY)
    base_url = overrides.pop("base_url", None)
    client_kwargs: dict[str, Any] = {"api_key": api_key, **overrides}
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs


def build_mem0_openai_config(app_settings: Settings | None = None) -> dict[str, Any]:
    """Build mem0 OpenAI provider config with optional Bifrost routing."""
    resolved_settings = _resolve_settings(app_settings)
    config: dict[str, Any] = {}
    if resolved_settings.BIFROST_ENABLED:
        config["api_key"] = resolved_settings.BIFROST_API_KEY
        config["openai_base_url"] = get_bifrost_openai_base_url(resolved_settings)
    return config


def make_chat_model(
    model: str,
    *,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    app_settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for LangChain chat models with optional settings injection."""
    return create_chat_model(
        model,
        configurable_fields=configurable_fields,
        app_settings=app_settings,
        **kwargs,
    )


def make_configurable_chat_model(
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ("model", "max_tokens", "api_key"),
    app_settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for configurable LangChain chat models."""
    return create_configurable_chat_model(
        configurable_fields=configurable_fields,
        app_settings=app_settings,
        **kwargs,
    )


def make_openai_chat_model(app_settings: Settings | None = None, **kwargs: Any) -> ChatOpenAI:
    """Factory for OpenAI-compatible chat models."""
    return create_openai_chat_model(app_settings=app_settings, **kwargs)
