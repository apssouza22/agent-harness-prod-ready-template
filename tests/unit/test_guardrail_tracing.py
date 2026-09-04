"""Unit tests for guardrail Langfuse span helpers."""

from unittest.mock import MagicMock

import pytest

from src.app.core.common.model.message import Message
from src.app.core.guardrails.tracing import get_guardrail_tracing_context, guardrail_span
from src.app.core.middleware.pipeline import AgentPipeline
from src.app.core.middleware.types import AgentContext, build_invoke_config


class _MockTracer:
    def __init__(self) -> None:
        self.client = MagicMock()
        self.created_spans: list[tuple[str, MagicMock]] = []

    def create_span(self, trace, name, input_data=None, metadata=None):
        span = MagicMock()
        self.created_spans.append((name, span))
        return span

    def end_span(self, span, output=None, metadata=None):
        span.end_output = output
        span.end_metadata = metadata


@pytest.mark.asyncio
async def test_guardrail_span_creates_child_span_under_active_trace():
    tracer = _MockTracer()
    root_trace = MagicMock()

    async def core_invoke(ctx: AgentContext):
        with guardrail_span(
            "guardrail_input_validation",
            input_data={"content_length": 2},
            ctx=ctx,
        ) as span_result:
            span_result["output"] = {"status": "passed"}
        return [Message(role="assistant", content="ok")]

    async def wrapped_invoke(inner_ctx: AgentContext):
        inner_ctx.metadata["langfuse_tracer"] = tracer
        inner_ctx.metadata["trace"] = root_trace
        return await core_invoke(inner_ctx)

    pipeline = AgentPipeline(middlewares=[], invoke_fn=wrapped_invoke)
    ctx = AgentContext(
        messages=[Message(role="user", content="hi")],
        session_id="session-1",
        user_id=1,
        config=build_invoke_config("session-1", 1, "test"),
        agent_name="test",
        metadata={"query": "hi"},
    )
    await pipeline.run(ctx)

    assert len(tracer.created_spans) == 1
    span_name, span = tracer.created_spans[0]
    assert span_name == "guardrail_input_validation"
    assert span.end_output == {"status": "passed"}


def test_get_guardrail_tracing_context_returns_none_without_active_ctx():
    tracer, trace, ctx = get_guardrail_tracing_context()
    assert tracer is None
    assert trace is None
    assert ctx is None


def test_guardrail_span_noop_when_tracing_unavailable():
    with guardrail_span("guardrail_input_validation") as span_result:
        span_result["output"] = {"status": "passed"}

    # No exception and no tracer required
