from langgraph.types import default_retry_on

from src.app.core.graph.config import FaultToleranceConfig
from src.app.core.graph.types import RetryPolicy, TimeoutPolicy

_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
)


def is_transient_error(exc: Exception) -> bool:
    """Return True when an exception is likely transient and safe to retry."""
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True

    return default_retry_on(exc)


def build_retry_policy(config: FaultToleranceConfig) -> RetryPolicy:
    """Build a RetryPolicy for LLM and orchestration nodes."""
    return RetryPolicy(
        initial_interval=config.initial_interval,
        backoff_factor=config.backoff_factor,
        max_interval=config.max_interval,
        max_attempts=config.max_attempts,
        jitter=config.jitter,
        retry_on=is_transient_error,
    )


def build_tool_retry_policy(config: FaultToleranceConfig) -> RetryPolicy:
    """Build a RetryPolicy for tool and retrieval nodes."""
    return RetryPolicy(
        initial_interval=config.initial_interval,
        backoff_factor=config.backoff_factor,
        max_interval=config.max_interval,
        max_attempts=config.max_attempts,
        jitter=config.jitter,
        retry_on=is_transient_error,
    )


def build_llm_timeout(config: FaultToleranceConfig) -> TimeoutPolicy:
    """Wall-clock and idle timeouts for LLM-heavy nodes."""
    return TimeoutPolicy(
        run_timeout=config.llm_run_timeout,
        idle_timeout=config.llm_idle_timeout,
    )


def build_tool_timeout(config: FaultToleranceConfig) -> TimeoutPolicy:
    """Wall-clock and idle timeouts for tool and search nodes."""
    return TimeoutPolicy(
        run_timeout=config.tool_run_timeout,
        idle_timeout=config.tool_idle_timeout,
    )
