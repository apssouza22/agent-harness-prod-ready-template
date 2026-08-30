"""Composable middleware for agent invocations.

Usage::

    from src.app.core.middleware import (
        AgentContext,
        AgentPipeline,
        ErrorHandlingMiddleware,
        LoggingMiddleware,
        LlmMetricsMiddleware,
        MemoryMiddleware,
        GuardrailMiddleware,
        SummarizationMiddleware,
        TrimLongMessagesMiddleware,
    )

    pipeline = AgentPipeline(
        middlewares=[LoggingMiddleware(), LlmMetricsMiddleware(), ErrorHandlingMiddleware(), MemoryMiddleware()],
        invoke_fn=agent.core_invoke,
    )
    result = await pipeline.run(ctx)
"""

from src.app.core.middleware.error_handling_middleware import ErrorHandlingMiddleware
from src.app.core.middleware.guardrail_middleware import GuardrailMiddleware
from src.app.core.middleware.llm_metrics_middleware import LlmMetricsMiddleware
from src.app.core.middleware.logging_middleware import LoggingMiddleware
from src.app.core.middleware.memory_middleware import MemoryMiddleware
from src.app.core.middleware.pipeline import (
    AgentPipeline,
    MiddlewareManager,
    get_active_middleware_manager,
    invoke_model,
)
from src.app.core.middleware.summarization_middleware import SummarizationMiddleware
from src.app.core.middleware.trim_long_messages_middleware import TrimLongMessagesMiddleware
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult, NextFn, build_invoke_config
