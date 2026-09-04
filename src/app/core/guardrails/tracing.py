"""Langfuse span helpers for guardrail observability."""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.app.core.langfuse.client import LangfuseTracer


@contextmanager
def guardrail_span(
    name: str,
    *,
    tracer: Optional[LangfuseTracer] = None,
    trace: Any | None = None,
    input_data: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Generator[dict[str, Any], None, None]:
    """Create a child span under the active request trace for a guardrail check.

    Yields a mutable dict; set ``output`` and optional ``metadata`` keys before exit
    to record span results via ``LangfuseTracer.end_span``.
    """
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
