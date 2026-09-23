"""Provider and model resolution helpers for long-term memory."""

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
