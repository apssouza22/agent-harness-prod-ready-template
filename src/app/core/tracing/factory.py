"""Backward-compatible Langfuse tracing factory helpers."""

from langfuse.langchain import CallbackHandler

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.langfuse.factory import make_langfuse_tracer


def make_langfuse_callback_handler() -> CallbackHandler:
    """Create a Langfuse callback handler via the shared tracer."""
    handler = make_langfuse_tracer().get_callback_handler()
    if handler is None:
        return CallbackHandler()
    return handler


def init_langfuse(app_settings: Settings | None = None) -> None:
    """Initialize and verify Langfuse client connectivity."""
    _ = app_settings or default_settings
    make_langfuse_tracer()


def shutdown_langfuse() -> None:
    """Flush and shut down the Langfuse client."""
    make_langfuse_tracer().shutdown()
