"""Langfuse tracer factory."""

from functools import lru_cache

from src.app.core.common.config import settings as default_settings
from src.app.core.common.logging import logger
from src.app.core.langfuse.client import LangfuseTracer


def get_settings():
    """Return application settings (imported lazily for test overrides)."""
    return default_settings


@lru_cache(maxsize=1)
def make_langfuse_tracer() -> LangfuseTracer:
    """Create a process-wide Langfuse tracer instance."""
    app_settings = get_settings()
    tracer = LangfuseTracer(app_settings)

    if tracer.client is None:
        return tracer

    try:
        if tracer.client.auth_check():
            logger.info("langfuse_auth_success")
        else:
            logger.error("langfuse_auth_failure")
    except Exception:
        logger.warning("langfuse_auth_check_failed", exc_info=True)

    return tracer
