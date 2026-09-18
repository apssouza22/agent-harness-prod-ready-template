"""Build mem0 configuration for OpenAI and AWS Bedrock providers."""

from typing import Any

from src.app.core.common.config import Settings

MEM0_OPENAI_PROVIDER = "openai"
MEM0_BEDROCK_PROVIDER = "aws_bedrock"

APP_OPENAI_PROVIDER = "openai"
APP_BEDROCK_PROVIDERS = frozenset({"bedrock", "bedrock_converse", "aws_bedrock"})


def _normalize_app_provider(provider: str) -> str:
    """Normalize configured provider names to app-level openai or bedrock."""
    normalized = provider.strip().lower()
    if normalized in APP_BEDROCK_PROVIDERS:
        return "bedrock"
    if normalized == APP_OPENAI_PROVIDER:
        return APP_OPENAI_PROVIDER
    raise ValueError(f"Unsupported memory provider: {provider}")


def _to_mem0_provider(app_provider: str) -> str:
    """Map app provider names to mem0 provider identifiers."""
    if app_provider == "bedrock":
        return MEM0_BEDROCK_PROVIDER
    return MEM0_OPENAI_PROVIDER


def normalize_memory_model(model: str) -> tuple[str | None, str]:
    """Strip LangChain-style prefixes and infer provider when present.

    Returns:
        A tuple of (provider_hint, bare_model_id). ``provider_hint`` is ``None``
        when the model string has no recognizable prefix.
    """
    if ":" not in model:
        return None, model

    prefix, bare_model = model.split(":", 1)
    normalized_prefix = prefix.lower()
    if normalized_prefix in {APP_OPENAI_PROVIDER}:
        return APP_OPENAI_PROVIDER, bare_model
    if normalized_prefix in APP_BEDROCK_PROVIDERS | {"bedrock"}:
        return "bedrock", bare_model
    return None, model


def resolve_memory_provider(
    configured_provider: str,
    model: str,
    *,
    fallback_provider: str,
) -> str:
    """Resolve the effective memory provider from settings and model prefix."""
    provider_hint, _ = normalize_memory_model(model)
    if provider_hint is not None:
        return provider_hint

    if configured_provider:
        return _normalize_app_provider(configured_provider)

    return _normalize_app_provider(fallback_provider)


def _build_bedrock_mem0_embedder_config(app_settings: Settings, model: str) -> dict[str, Any]:
    """Build mem0 aws_bedrock embedder config (BaseEmbedderConfig field subset)."""
    config: dict[str, Any] = {"model": model}
    if app_settings.AWS_REGION:
        config["aws_region"] = app_settings.AWS_REGION
    if app_settings.AWS_ACCESS_KEY_ID:
        config["aws_access_key_id"] = app_settings.AWS_ACCESS_KEY_ID
    if app_settings.AWS_SECRET_ACCESS_KEY:
        config["aws_secret_access_key"] = app_settings.AWS_SECRET_ACCESS_KEY
    return config


def build_mem0_llm_config(app_settings: Settings) -> dict[str, Any]:
    """Build the mem0 LLM configuration section."""
    provider = resolve_memory_provider(
        app_settings.LONG_TERM_MEMORY_LLM_PROVIDER,
        app_settings.LONG_TERM_MEMORY_MODEL,
        fallback_provider=app_settings.DEFAULT_LLM_PROVIDER,
    )
    _, model = normalize_memory_model(app_settings.LONG_TERM_MEMORY_MODEL)
    mem0_provider = _to_mem0_provider(provider)

    if mem0_provider == MEM0_BEDROCK_PROVIDER:
        # mem0's LlmFactory instantiates BaseLlmConfig for aws_bedrock; AWSBedrockLLM
        # reads credentials from environment variables after that conversion.
        config = {"model": model}
    else:
        config = {"model": model}
        if app_settings.OPENAI_API_KEY:
            config["api_key"] = app_settings.OPENAI_API_KEY

    return {"provider": mem0_provider, "config": config}


def build_mem0_embedder_config(app_settings: Settings) -> dict[str, Any]:
    """Build the mem0 embedder configuration section."""
    provider = resolve_memory_provider(
        app_settings.LONG_TERM_MEMORY_EMBEDDER_PROVIDER,
        app_settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
        fallback_provider=app_settings.LONG_TERM_MEMORY_LLM_PROVIDER or app_settings.DEFAULT_LLM_PROVIDER,
    )
    _, model = normalize_memory_model(app_settings.LONG_TERM_MEMORY_EMBEDDER_MODEL)
    mem0_provider = _to_mem0_provider(provider)

    if mem0_provider == MEM0_BEDROCK_PROVIDER:
        config = _build_bedrock_mem0_embedder_config(app_settings, model)
    else:
        config = {"model": model}
        if app_settings.OPENAI_API_KEY:
            config["api_key"] = app_settings.OPENAI_API_KEY

    return {"provider": mem0_provider, "config": config}
