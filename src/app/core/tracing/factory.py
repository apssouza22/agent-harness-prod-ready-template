"""Langfuse tracing factory."""

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.common.logging import logger


def make_langfuse_callback_handler() -> CallbackHandler:
    """Create a Langfuse callback handler for tracking LLM interactions.

    Returns:
        CallbackHandler: Configured Langfuse callback handler.
    """
    return CallbackHandler()


def init_langfuse(app_settings: Settings | None = None) -> None:
    """Initialize and verify Langfuse client connectivity.

    Args:
        app_settings: Application settings. Falls back to the module-level singleton.
    """
    _ = app_settings or default_settings
    langfuse = get_client()

    if langfuse.auth_check():
        logger.info("langfuse_auth_success")
        return

    logger.error("langfuse_auth_failure")


def shutdown_langfuse() -> None:
    """Flush and shut down the Langfuse client."""
    get_client().shutdown()
    logger.info("langfuse_shutdown_complete")
