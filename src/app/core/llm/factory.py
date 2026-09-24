"""Centralized LLM factory with optional Bifrost API gateway routing.

When BIFROST_ENABLED is true, LangChain chat models are pointed at the Bifrost
/langchain proxy and OpenAI-compatible clients use the /v1 endpoint. Provider
API keys are managed by Bifrost instead of the application.

When DEFAULT_LLM_PROVIDER=bedrock, models route through langchain-aws
ChatBedrockConverse using AWS credentials instead of API keys.
"""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.common.logging import logger


settings = default_settings

BifrostAgent = Literal["agent_1", "agent_2"]

OPENAI_ONLY_KWARGS = frozenset({"reasoning"})
BEDROCK_UNSUPPORTED_INFERENCE_KWARGS = frozenset({"temperature", "top_p"})
BEDROCK_CONVERSE_PREFIX = "bedrock_converse"
BEDROCK_LEGACY_PREFIX = "bedrock"
BEDROCK_MIN_MAX_TOKENS = 16
BEDROCK_REGIONAL_PREFIXES = (
    "eu.",
    "us.",
    "us-gov.",
    "apac.",
    "sa.",
    "amer.",
    "global.",
    "jp.",
    "au.",
)


def _resolve_settings(app_settings: Settings | None = None) -> Settings:
    return app_settings or settings


def resolve_bifrost_virtual_key(
    *,
    bifrost_agent: BifrostAgent | None = None,
    app_settings: Settings | None = None,
) -> str:
    """Resolve the Bifrost virtual key for an agent tier or global fallback."""
    resolved_settings = _resolve_settings(app_settings)
    if bifrost_agent == "agent_1" and resolved_settings.BIFROST_API_KEY_AGENT_1:
        return resolved_settings.BIFROST_API_KEY_AGENT_1
    if bifrost_agent == "agent_2" and resolved_settings.BIFROST_API_KEY_AGENT_2:
        return resolved_settings.BIFROST_API_KEY_AGENT_2
    return resolved_settings.BIFROST_VIRTUAL_KEY or ""


def get_bifrost_langchain_base_url(app_settings: Settings | None = None) -> str:
    """Return the Bifrost LangChain proxy base URL."""
    resolved_settings = _resolve_settings(app_settings)
    return resolved_settings.BIFROST_BASE_URL.rstrip("/")


def get_bifrost_openai_base_url(app_settings: Settings | None = None) -> str:
    """Return the Bifrost OpenAI-compatible base URL for direct OpenAI clients."""
    resolved_settings = _resolve_settings(app_settings)
    return resolved_settings.BIFROST_OPENAI_BASE_URL.rstrip("/")


def get_bifrost_default_headers(
    app_settings: Settings | None = None,
    *,
    bifrost_agent: BifrostAgent | None = None,
) -> dict[str, str]:
    """Return optional Bifrost governance headers (e.g. virtual key)."""
    virtual_key = resolve_bifrost_virtual_key(bifrost_agent=bifrost_agent, app_settings=app_settings)
    if not virtual_key:
        return {}
    return {"x-bf-vk": virtual_key}


def _parse_model_provider(model_name: str) -> str | None:
    """Extract the provider prefix from a model identifier, if present."""
    if ":" not in model_name:
        return None
    return model_name.split(":", 1)[0].lower()


def _normalize_bedrock_model_identifier(model_name: str) -> str:
    """Map legacy ``bedrock:`` identifiers to the Converse API prefix."""
    lower = model_name.lower()
    if lower.startswith(f"{BEDROCK_CONVERSE_PREFIX}:"):
        return model_name
    if lower.startswith(f"{BEDROCK_LEGACY_PREFIX}:"):
        return f"{BEDROCK_CONVERSE_PREFIX}:{model_name.split(':', 1)[1]}"
    return model_name


def _resolve_bedrock_langchain_prefix(provider: str) -> str:
    """Return the LangChain provider prefix for AWS Bedrock routing."""
    if provider == BEDROCK_LEGACY_PREFIX:
        return BEDROCK_CONVERSE_PREFIX
    return provider


def resolve_model_identifier(
    model: str | None = None,
    provider: str | None = None,
    app_settings: Settings | None = None,
) -> str:
    """Build a provider-prefixed model identifier for LangChain init_chat_model.

    If ``model`` already contains a provider prefix (e.g. ``bedrock_converse:...``),
    it is returned unchanged. Legacy ``bedrock:`` prefixes are normalized to
    ``bedrock_converse:``. Otherwise the provider is taken from ``provider``
    or ``DEFAULT_LLM_PROVIDER``.
    """
    resolved_settings = _resolve_settings(app_settings)
    model_name = model or resolved_settings.DEFAULT_LLM_MODEL
    if ":" in model_name:
        return _normalize_bedrock_model_identifier(model_name)
    resolved_provider = provider or resolved_settings.DEFAULT_LLM_PROVIDER
    langchain_provider = _resolve_bedrock_langchain_prefix(resolved_provider)
    return f"{langchain_provider}:{model_name}"


def _clamp_bedrock_max_tokens(kwargs: dict[str, Any]) -> None:
    """Raise Bedrock max_tokens to the provider minimum when needed."""
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is not None and max_tokens < BEDROCK_MIN_MAX_TOKENS:
        logger.debug(
            "bedrock_max_tokens_clamped",
            requested_max_tokens=max_tokens,
            min_max_tokens=BEDROCK_MIN_MAX_TOKENS,
        )
        kwargs["max_tokens"] = BEDROCK_MIN_MAX_TOKENS


def build_bedrock_client_kwargs(app_settings: Settings | None = None) -> dict[str, Any]:
    """Build AWS client kwargs for ChatBedrockConverse."""
    resolved_settings = _resolve_settings(app_settings)
    kwargs: dict[str, Any] = {"region_name": resolved_settings.AWS_REGION}

    if resolved_settings.AWS_PROFILE:
        kwargs["credentials_profile_name"] = resolved_settings.AWS_PROFILE
    if resolved_settings.AWS_ACCESS_KEY_ID:
        kwargs["aws_access_key_id"] = resolved_settings.AWS_ACCESS_KEY_ID
    if resolved_settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_secret_access_key"] = resolved_settings.AWS_SECRET_ACCESS_KEY
    if resolved_settings.AWS_SESSION_TOKEN:
        kwargs["aws_session_token"] = resolved_settings.AWS_SESSION_TOKEN

    return kwargs


def extract_bedrock_base_model_id(model_id: str) -> str:
    """Return the Bedrock base model id without a regional prefix."""
    if model_id.startswith(BEDROCK_REGIONAL_PREFIXES):
        return model_id.partition(".")[2]
    return model_id


def bedrock_model_supports_temperature(model_name: str) -> bool:
    """Return False when langchain-aws marks the model as temperature-incompatible."""
    provider = _parse_model_provider(model_name)
    bare_model = model_name.split(":", 1)[1] if provider else model_name
    base_model = extract_bedrock_base_model_id(bare_model)

    try:
        from langchain_aws.chat_models.bedrock_converse import _get_default_model_profile
    except ImportError:
        return True

    profile = _get_default_model_profile(base_model)
    if not profile:
        return True
    return profile.get("temperature") is not False


def build_bedrock_converse_inference_config(
    model_name: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Build Bedrock Converse inferenceConfig with unsupported fields removed."""
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    if bedrock_model_supports_temperature(model_name):
        inference_config["temperature"] = temperature
        inference_config["topP"] = top_p
    return inference_config


def filter_provider_kwargs(model_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove provider-incompatible kwargs before model initialization."""
    provider = _parse_model_provider(model_name)
    if provider == "openai":
        return kwargs

    filtered = {key: value for key, value in kwargs.items() if key not in OPENAI_ONLY_KWARGS}
    if provider in {BEDROCK_LEGACY_PREFIX, BEDROCK_CONVERSE_PREFIX} and not bedrock_model_supports_temperature(
        model_name
    ):
        filtered = {
            key: value for key, value in filtered.items() if key not in BEDROCK_UNSUPPORTED_INFERENCE_KWARGS
        }

    removed = set(kwargs) - set(filtered)
    if removed:
        logger.debug("provider_kwargs_filtered", model=model_name, removed_keys=sorted(removed))
    return filtered


def resolve_api_key_for_model(model_name: str | None = None, app_settings: Settings | None = None) -> str:
    """Resolve the API key to pass to LangChain for the given model."""
    resolved_settings = _resolve_settings(app_settings)
    if resolved_settings.BIFROST_ENABLED:
        return resolved_settings.BIFROST_API_KEY

    if model_name:
        normalized = model_name.lower()
        if normalized.startswith(f"{BEDROCK_CONVERSE_PREFIX}:") or normalized.startswith(f"{BEDROCK_LEGACY_PREFIX}:"):
            return ""
        if normalized.startswith("openai:"):
            return resolved_settings.OPENAI_API_KEY
        if normalized.startswith("anthropic:"):
            return resolved_settings.ANTHROPIC_API_KEY or ""
        if normalized.startswith("google"):
            return resolved_settings.GOOGLE_API_KEY or ""

    return resolved_settings.OPENAI_API_KEY


def build_chat_model_kwargs(
    app_settings: Settings | None = None,
    *,
    bifrost_agent: BifrostAgent | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build kwargs for LangChain chat model initialization."""
    resolved_settings = _resolve_settings(app_settings)
    kwargs: dict[str, Any] = dict(overrides)

    model_name = kwargs.get("model", "")
    if model_name:
        model_name = _normalize_bedrock_model_identifier(model_name)
        kwargs["model"] = model_name
    provider = _parse_model_provider(model_name)

    if resolved_settings.BIFROST_ENABLED:
        kwargs["base_url"] = get_bifrost_langchain_base_url(resolved_settings)
        kwargs["api_key"] = resolved_settings.BIFROST_API_KEY
        bifrost_headers = get_bifrost_default_headers(resolved_settings, bifrost_agent=bifrost_agent)
        if bifrost_headers:
            existing_headers = kwargs.get("default_headers", {})
            kwargs["default_headers"] = {**existing_headers, **bifrost_headers}
        logger.debug(
            "bifrost_chat_model_routing_enabled",
            base_url=kwargs["base_url"],
            bifrost_agent=bifrost_agent,
            has_virtual_key=bool(
                resolve_bifrost_virtual_key(bifrost_agent=bifrost_agent, app_settings=resolved_settings)
            ),
        )
        return filter_provider_kwargs(model_name, kwargs)

    if provider in {BEDROCK_LEGACY_PREFIX, BEDROCK_CONVERSE_PREFIX}:
        kwargs.update(build_bedrock_client_kwargs(resolved_settings))
        kwargs.pop("api_key", None)
        _clamp_bedrock_max_tokens(kwargs)
        logger.debug(
            "bedrock_chat_model_routing_enabled",
            model=model_name,
            region_name=kwargs.get("region_name"),
            has_profile=bool(resolved_settings.AWS_PROFILE),
        )
        return filter_provider_kwargs(model_name, kwargs)

    if "api_key" not in kwargs and model_name:
        api_key = resolve_api_key_for_model(model_name, resolved_settings)
        if api_key:
            kwargs["api_key"] = api_key

    return filter_provider_kwargs(model_name, kwargs)


def make_chat_model(
    model: str,
    fallbacks: list[str] | None = None,
    *,
    app_settings: Settings | None = None,
    bifrost_agent: BifrostAgent | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain chat model with optional fallbacks.

    Instantiates the correct provider model based on the ``model`` identifier
    (e.g. ``openai:gpt-4o``, ``bedrock_converse:us.anthropic.claude-3-7-sonnet-20250219-v1:0``).
    When ``fallbacks`` are provided, each fallback model is instantiated with the
    same kwargs and attached via ``with_fallbacks``.

    Args:
        model: Provider-prefixed model identifier passed to ``init_chat_model``.
        fallbacks: Optional list of fallback model identifiers.
        app_settings: Optional settings override for provider routing and API keys.
        bifrost_agent: Optional Bifrost agent tier for per-agent virtual keys.
        configurable_fields: Optional fields to make runtime-configurable.
        **kwargs: Additional kwargs forwarded to ``init_chat_model``.

    Returns:
        A configured LangChain chat model, optionally wrapped with fallbacks.
    """
    model_kwargs = build_chat_model_kwargs(
        app_settings,
        bifrost_agent=bifrost_agent,
        model=model,
        **kwargs,
    )
    if configurable_fields is not None:
        chat_model = init_chat_model(configurable_fields=configurable_fields, **model_kwargs)
    else:
        chat_model = init_chat_model(**model_kwargs)

    if not fallbacks:
        return chat_model

    fallback_models = [
        make_chat_model(
            fallback_model,
            app_settings=app_settings,
            bifrost_agent=bifrost_agent,
            **kwargs,
        )
        for fallback_model in fallbacks
    ]
    return chat_model.with_fallbacks(fallback_models)


def build_openai_client_kwargs(
    app_settings: Settings | None = None,
    *,
    bifrost_agent: BifrostAgent | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build kwargs for OpenAI SDK clients (evaluations, etc.)."""
    resolved_settings = _resolve_settings(app_settings)
    if resolved_settings.BIFROST_ENABLED:
        virtual_key = resolve_bifrost_virtual_key(bifrost_agent=bifrost_agent, app_settings=resolved_settings)
        return {
            "api_key": virtual_key or resolved_settings.BIFROST_API_KEY,
            "base_url": get_bifrost_openai_base_url(resolved_settings),
            **overrides,
        }

    api_key = overrides.pop("api_key", resolved_settings.OPENAI_API_KEY)
    base_url = overrides.pop("base_url", None)
    client_kwargs: dict[str, Any] = {"api_key": api_key, **overrides}
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs


def build_mem0_openai_config(
    app_settings: Settings | None = None,
    *,
    bifrost_agent: BifrostAgent | None = None,
) -> dict[str, Any]:
    """Build mem0 OpenAI provider config with optional Bifrost routing."""
    resolved_settings = _resolve_settings(app_settings)
    config: dict[str, Any] = {}
    if resolved_settings.BIFROST_ENABLED:
        virtual_key = resolve_bifrost_virtual_key(bifrost_agent=bifrost_agent, app_settings=resolved_settings)
        config["api_key"] = virtual_key or resolved_settings.BIFROST_API_KEY
        config["openai_base_url"] = get_bifrost_openai_base_url(resolved_settings)
    return config
