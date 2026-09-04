"""Text-to-SQL agent factory."""

from src.app.agents.text_to_sql.text_sql_agent import (
    TextSQLDeepAgent,
    build_text_to_sql_trace_metadata,
    build_text_to_sql_trace_output,
)
from src.app.core.common.config import settings
from src.app.core.langfuse import LangfuseTracer, LangfuseTracingMiddleware
from src.app.core.guardrails import GuardrailMiddleware
from src.app.core.metrics import LlmMetricsMiddleware
from src.app.core.middleware import ErrorHandlingMiddleware, LoggingMiddleware


async def make_text_to_sql_agent(langfuse_tracer: LangfuseTracer | None = None) -> TextSQLDeepAgent:
    """Create a text-to-SQL deep agent.

    Args:
        langfuse_tracer: Optional Langfuse tracer for observability.

    Returns:
        TextSQLDeepAgent: Text-to-SQL agent instance.
    """
    middlewares = [
        LangfuseTracingMiddleware(
            langfuse_tracer=langfuse_tracer,
            trace_name="text_to_sql_request",
            environment=settings.ENVIRONMENT.value,
            build_trace_metadata=build_text_to_sql_trace_metadata,
            build_trace_output=build_text_to_sql_trace_output,
        ),
        LoggingMiddleware(),
        LlmMetricsMiddleware(),
        ErrorHandlingMiddleware(),
        GuardrailMiddleware(langfuse_tracer=langfuse_tracer),
    ]
    return TextSQLDeepAgent(name="Text-to-SQL", middlewares=middlewares)
