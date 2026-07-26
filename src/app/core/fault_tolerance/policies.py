"""Retry and timeout policy factories for LangGraph nodes."""

from langgraph.types import RetryPolicy, TimeoutPolicy

from src.app.core.common.config import settings


def get_llm_retry_policy() -> RetryPolicy:
    """Retry policy for LLM-calling graph nodes."""
    return RetryPolicy(
        max_attempts=settings.GRAPH_LLM_RETRY_MAX_ATTEMPTS,
        initial_interval=settings.GRAPH_RETRY_INITIAL_INTERVAL,
        backoff_factor=settings.GRAPH_RETRY_BACKOFF_FACTOR,
        max_interval=settings.GRAPH_RETRY_MAX_INTERVAL,
        jitter=settings.GRAPH_RETRY_JITTER,
    )


def get_tool_retry_policy() -> RetryPolicy:
    """Retry policy for tool-execution graph nodes."""
    return RetryPolicy(
        max_attempts=settings.GRAPH_TOOL_RETRY_MAX_ATTEMPTS,
        initial_interval=settings.GRAPH_RETRY_INITIAL_INTERVAL,
        backoff_factor=settings.GRAPH_RETRY_BACKOFF_FACTOR,
        max_interval=settings.GRAPH_RETRY_MAX_INTERVAL,
        jitter=settings.GRAPH_RETRY_JITTER,
    )


def get_llm_timeout_policy() -> TimeoutPolicy:
    """Timeout policy for LLM-calling graph nodes."""
    return TimeoutPolicy(
        run_timeout=settings.GRAPH_LLM_RUN_TIMEOUT,
        idle_timeout=settings.GRAPH_LLM_IDLE_TIMEOUT,
    )


def get_tool_timeout_policy() -> TimeoutPolicy:
    """Timeout policy for tool-execution graph nodes."""
    return TimeoutPolicy(
        run_timeout=settings.GRAPH_TOOL_RUN_TIMEOUT,
        idle_timeout=settings.GRAPH_TOOL_IDLE_TIMEOUT,
    )
