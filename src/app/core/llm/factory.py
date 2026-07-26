"""Centralized LLM factory with optional Bifrost API gateway routing.

When BIFROST_ENABLED is true, LangChain chat models are pointed at the Bifrost
/langchain proxy and OpenAI-compatible clients use the /v1 endpoint. Provider
API keys are managed by Bifrost instead of the application.
"""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.app.core.common.config import settings
from src.app.core.common.logging import logger


def get_bifrost_langchain_base_url() -> str:
    """Return the Bifrost LangChain proxy base URL."""
    return settings.BIFROST_BASE_URL.rstrip("/")


def get_bifrost_openai_base_url() -> str:
    """Return the Bifrost OpenAI-compatible base URL for direct OpenAI clients."""
    return settings.BIFROST_OPENAI_BASE_URL.rstrip("/")


def get_bifrost_default_headers() -> dict[str, str]:
    """Return optional Bifrost governance headers (e.g. virtual key)."""
    if not settings.BIFROST_VIRTUAL_KEY:
        return {}
    return {"x-bf-vk": settings.BIFROST_VIRTUAL_KEY}


def resolve_api_key_for_model(model_name: str | None = None) -> str:
    """Resolve the API key to pass to LangChain for the given model."""
    if settings.BIFROST_ENABLED:
        return settings.BIFROST_API_KEY

    if model_name:
        normalized = model_name.lower()
        if normalized.startswith("openai:"):
            return settings.OPENAI_API_KEY
        if normalized.startswith("anthropic:"):
            return settings.ANTHROPIC_API_KEY or ""
        if normalized.startswith("google"):
            return settings.GOOGLE_API_KEY or ""

    return settings.OPENAI_API_KEY


def build_chat_model_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build kwargs for LangChain chat model initialization."""
    kwargs: dict[str, Any] = dict(overrides)

    if settings.BIFROST_ENABLED:
        kwargs["base_url"] = get_bifrost_langchain_base_url()
        kwargs["api_key"] = settings.BIFROST_API_KEY
        bifrost_headers = get_bifrost_default_headers()
        if bifrost_headers:
            existing_headers = kwargs.get("default_headers", {})
            kwargs["default_headers"] = {**existing_headers, **bifrost_headers}
        logger.debug(
            "bifrost_chat_model_routing_enabled",
            base_url=kwargs["base_url"],
            has_virtual_key=bool(settings.BIFROST_VIRTUAL_KEY),
        )
        return kwargs

    if "api_key" not in kwargs and "model" in kwargs:
        kwargs["api_key"] = resolve_api_key_for_model(kwargs["model"])

    return kwargs


def create_chat_model(
    model: str,
    *,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain chat model, routing through Bifrost when enabled."""
    model_kwargs = build_chat_model_kwargs(model=model, **kwargs)
    if configurable_fields is not None:
        return init_chat_model(configurable_fields=configurable_fields, **model_kwargs)
    return init_chat_model(**model_kwargs)


def create_configurable_chat_model(
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ("model", "max_tokens", "api_key"),
    **kwargs: Any,
) -> BaseChatModel:
    """Create a configurable chat model with Bifrost defaults applied."""
    model_kwargs = build_chat_model_kwargs(**kwargs)
    return init_chat_model(configurable_fields=configurable_fields, **model_kwargs)


def create_openai_chat_model(**kwargs: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance, routing through Bifrost when enabled."""
    model_kwargs = build_chat_model_kwargs(**kwargs)
    return ChatOpenAI(**model_kwargs)


def build_openai_client_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build kwargs for OpenAI SDK clients (mem0, evaluations, etc.)."""
    if settings.BIFROST_ENABLED:
        return {
            "api_key": settings.BIFROST_API_KEY,
            "base_url": get_bifrost_openai_base_url(),
            **overrides,
        }

    api_key = overrides.pop("api_key", settings.OPENAI_API_KEY)
    base_url = overrides.pop("base_url", None)
    client_kwargs: dict[str, Any] = {"api_key": api_key, **overrides}
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs


def build_mem0_openai_config() -> dict[str, Any]:
    """Build mem0 OpenAI provider config with optional Bifrost routing."""
    config: dict[str, Any] = {}
    if settings.BIFROST_ENABLED:
        config["api_key"] = settings.BIFROST_API_KEY
        config["openai_base_url"] = get_bifrost_openai_base_url()
    return config
