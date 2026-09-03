from pydantic import BaseModel, Field

from src.app.core.common.config import settings


class FaultToleranceConfig(BaseModel):
    """LangGraph fault-tolerance settings (retries, timeouts, error handlers).

    Based on LangGraph RetryPolicy, TimeoutPolicy, and error_handler primitives.
    See: https://www.langchain.com/blog/fault-tolerance-in-langgraph
    """

    enabled: bool = True
    max_attempts: int = Field(default_factory=lambda: settings.GRAPH_LLM_RETRY_MAX_ATTEMPTS)
    initial_interval: float = Field(default_factory=lambda: settings.GRAPH_RETRY_INITIAL_INTERVAL)
    backoff_factor: float = Field(default_factory=lambda: settings.GRAPH_RETRY_BACKOFF_FACTOR)
    max_interval: float = Field(default_factory=lambda: settings.GRAPH_RETRY_MAX_INTERVAL)
    jitter: bool = Field(default_factory=lambda: settings.GRAPH_RETRY_JITTER)
    llm_run_timeout: float = Field(default_factory=lambda: settings.GRAPH_LLM_RUN_TIMEOUT)
    llm_idle_timeout: float = Field(default_factory=lambda: settings.GRAPH_LLM_IDLE_TIMEOUT)
    tool_run_timeout: float = Field(default_factory=lambda: settings.GRAPH_TOOL_RUN_TIMEOUT)
    tool_idle_timeout: float = Field(default_factory=lambda: settings.GRAPH_TOOL_IDLE_TIMEOUT)
