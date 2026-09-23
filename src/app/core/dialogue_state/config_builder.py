"""Provider and model resolution helpers for dialogue state tracking."""

APP_OPENAI_PROVIDER = "openai"
APP_BEDROCK_PROVIDERS = frozenset({"bedrock", "bedrock_converse", "aws_bedrock"})


def _normalize_app_provider(provider: str) -> str:
    """Normalize configured provider names to app-level openai or bedrock."""
    normalized = provider.strip().lower()
    if normalized in APP_BEDROCK_PROVIDERS:
        return "bedrock"
    if normalized == APP_OPENAI_PROVIDER:
        return APP_OPENAI_PROVIDER
    raise ValueError(f"Unsupported dialogue state provider: {provider}")


def normalize_dialogue_state_model(model: str) -> tuple[str | None, str]:
    """Strip LangChain-style prefixes and infer provider when present."""
    if ":" not in model:
        return None, model

    prefix, bare_model = model.split(":", 1)
    normalized_prefix = prefix.lower()
    if normalized_prefix in {APP_OPENAI_PROVIDER}:
        return APP_OPENAI_PROVIDER, bare_model
    if normalized_prefix in APP_BEDROCK_PROVIDERS | {"bedrock"}:
        return "bedrock", bare_model
    return None, model


def resolve_dialogue_state_provider(
    configured_provider: str,
    model: str,
    *,
    fallback_provider: str,
) -> str:
    """Resolve the effective dialogue state provider from settings and model prefix."""
    provider_hint, _ = normalize_dialogue_state_model(model)
    if provider_hint is not None:
        return provider_hint

    if configured_provider:
        return _normalize_app_provider(configured_provider)

    return _normalize_app_provider(fallback_provider)
