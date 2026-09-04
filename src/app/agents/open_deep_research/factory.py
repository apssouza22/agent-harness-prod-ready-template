"""Deep research agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.open_deep_research.agent_deep_research import (
    DeepResearchAgent,
    build_deep_research_trace_metadata,
    build_deep_research_trace_output,
)
from src.app.core.common.config import settings
from src.app.core.langfuse import LangfuseTracer, LangfuseTracingMiddleware
from src.app.core.middleware import (
    ErrorHandlingMiddleware,
    GuardrailMiddleware,
    LlmMetricsMiddleware,
    LoggingMiddleware,
    MemoryMiddleware,
)


async def make_deep_research_agent(
    checkpointer: AsyncPostgresSaver | None,
    langfuse_tracer: LangfuseTracer | None = None,
) -> DeepResearchAgent:
    """Create and compile a deep research agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.
        langfuse_tracer: Optional Langfuse tracer for observability.

    Returns:
        DeepResearchAgent: Compiled deep research agent instance.
    """
    middlewares = [
        LangfuseTracingMiddleware(
            langfuse_tracer=langfuse_tracer,
            trace_name="deep_research_request",
            environment=settings.ENVIRONMENT.value,
            build_trace_metadata=build_deep_research_trace_metadata,
            build_trace_output=build_deep_research_trace_output,
        ),
        LoggingMiddleware(),
        GuardrailMiddleware(langfuse_tracer=langfuse_tracer),
        LlmMetricsMiddleware(),
        ErrorHandlingMiddleware(),
        MemoryMiddleware(),
    ]
    agent = DeepResearchAgent("Deep Research", checkpointer, middlewares=middlewares)
    await agent.compile()
    return agent
