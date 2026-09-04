"""Langfuse span helpers for guardrail observability."""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.app.core.langfuse.client import LangfuseTracer
from src.app.core.middleware.pipeline import get_active_middleware_manager
from src.app.core.middleware.types import AgentContext


def get_guardrail_tracing_context(
    ctx: AgentContext | None = None,
) -> tuple[Optional[LangfuseTracer], Any, Optional[AgentContext]]:
    """Resolve tracer, root observation, and context for guardrail spans."""
    active_ctx = ctx
    if active_ctx is None:
        manager = get_active_middleware_manager()
        if manager:
            active_ctx = manager.active_ctx

    if active_ctx is None:
        return None, None, None

    tracer = active_ctx.metadata.get("langfuse_tracer")
    trace = active_ctx.metadata.get("trace")
    return tracer, trace, active_ctx


@contextmanager
def guardrail_span(
    name: str,
    *,
    input_data: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
    ctx: AgentContext | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Create a child span under the active request trace for a guardrail check.

    Yields a mutable dict; set ``output`` and optional ``metadata`` keys before exit
    to record span results via ``LangfuseTracer.end_span``.
    """
    tracer, trace, _ = get_guardrail_tracing_context(ctx)
    span = None
    span_result: dict[str, Any] = {}

    if tracer and trace:
        span = tracer.create_span(trace, name, input_data=input_data, metadata=metadata)

    try:
        yield span_result
    finally:
        if tracer and span:
            tracer.end_span(
                span,
                output=span_result.get("output"),
                metadata=span_result.get("metadata"),
            )
