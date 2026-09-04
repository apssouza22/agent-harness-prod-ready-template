"""Composable middleware for agent invocations.

Usage::

    from src.app.core.middleware import (
        AgentContext,
        AgentPipeline,
        ErrorHandlingMiddleware,
        LoggingMiddleware,
    )
    from src.app.core.memory import MemoryMiddleware
    from src.app.core.metrics import LlmMetricsMiddleware

    pipeline = AgentPipeline(
        middlewares=[LoggingMiddleware(), LlmMetricsMiddleware(), ErrorHandlingMiddleware(), MemoryMiddleware()],
        invoke_fn=agent.core_invoke,
    )
    result = await pipeline.run(ctx)
"""

from src.app.core.middleware.error_handling_middleware import ErrorHandlingMiddleware
from src.app.core.middleware.logging_middleware import LoggingMiddleware
from src.app.core.middleware.pipeline import (
    AgentPipeline,
    MiddlewareManager,
    get_active_middleware_manager,
    invoke_model,
    middleware_tool_wrappers,
)
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult, NextFn, build_invoke_config

__all__ = [
    "AgentContext",
    "AgentMiddleware",
    "AgentPipeline",
    "ErrorHandlingMiddleware",
    "InvokeResult",
    "LoggingMiddleware",
    "MiddlewareManager",
    "NextFn",
    "build_invoke_config",
    "get_active_middleware_manager",
    "invoke_model",
    "middleware_tool_wrappers",
]
