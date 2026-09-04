"""Runtime holder for the active Langfuse callback handler.

Set once during application startup so non-request code (e.g. middleware
invoke config builders) can access the handler without importing globals.
"""

from langfuse.langchain import CallbackHandler

_active_langfuse_callback_handler: CallbackHandler | None = None


def set_active_langfuse_callback_handler(handler: CallbackHandler) -> None:
    """Register the process-wide Langfuse callback handler."""
    global _active_langfuse_callback_handler
    _active_langfuse_callback_handler = handler


def get_active_langfuse_callback_handler() -> CallbackHandler | None:
    """Return the active Langfuse callback handler, if startup has completed."""
    return _active_langfuse_callback_handler


def clear_active_langfuse_callback_handler() -> None:
    """Clear the active handler during shutdown."""
    global _active_langfuse_callback_handler
    _active_langfuse_callback_handler = None
