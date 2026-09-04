"""Unit tests for LangfuseTracingMiddleware."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.app.core.common.model.message import Message
from src.app.core.middleware.langfuse_tracing_middleware import LangfuseTracingMiddleware
from src.app.core.middleware.pipeline import AgentPipeline
from src.app.core.middleware.types import AgentContext, build_invoke_config


class _MockTracer:
    def __init__(self) -> None:
        self.client = MagicMock()
        self.flush = MagicMock()
        self._trace_id = "trace-123"

    @contextmanager
    def trace_agent_request(self, name, **kwargs):
        observation = MagicMock()
        observation.trace_id = self._trace_id
        yield observation

    def get_callback_handler(self):
        return MagicMock(name="callback_handler")

    def get_trace_id(self, trace=None):
        if trace is not None:
            return getattr(trace, "trace_id", self._trace_id)
        return self._trace_id


@pytest.mark.asyncio
async def test_langfuse_tracing_middleware_sets_trace_id_and_callbacks():
    tracer = _MockTracer()
    middleware = LangfuseTracingMiddleware(
        langfuse_tracer=tracer,
        trace_name="test_agent_request",
        environment="test",
    )

    async def core_invoke(ctx: AgentContext):
        ctx.metadata["graph_result"] = {"messages": []}
        return [Message(role="assistant", content="hello")]

    pipeline = AgentPipeline(middlewares=[middleware], invoke_fn=core_invoke)
    ctx = AgentContext(
        messages=[Message(role="user", content="hi")],
        session_id="session-1",
        user_id=42,
        config=build_invoke_config("session-1", 42, "test-agent"),
        agent_name="test-agent",
        metadata={
            "query": "hi",
            "user_id": 42,
            "trace_metadata": {"service": "test"},
        },
    )

    result = await pipeline.run(ctx)

    assert result[0].content == "hello"
    assert ctx.metadata["trace_id"] == "trace-123"
    assert len(ctx.config["callbacks"]) == 1
    tracer.flush.assert_called_once()


@pytest.mark.asyncio
async def test_langfuse_tracing_middleware_noop_when_tracer_disabled():
    middleware = LangfuseTracingMiddleware(langfuse_tracer=None, trace_name="test_agent_request")

    async def core_invoke(_ctx: AgentContext):
        return [Message(role="assistant", content="ok")]

    pipeline = AgentPipeline(middlewares=[middleware], invoke_fn=core_invoke)
    ctx = AgentContext(
        messages=[Message(role="user", content="hi")],
        session_id="session-1",
        user_id=None,
        config=build_invoke_config("session-1", None, "test-agent"),
        agent_name="test-agent",
        metadata={"query": "hi"},
    )

    result = await pipeline.run(ctx)

    assert result[0].content == "ok"
    assert "trace_id" not in ctx.metadata


@pytest.mark.asyncio
async def test_langfuse_tracing_middleware_records_error_on_failure():
    tracer = _MockTracer()
    middleware = LangfuseTracingMiddleware(
        langfuse_tracer=tracer,
        trace_name="test_agent_request",
    )

    async def core_invoke(_ctx: AgentContext):
        raise RuntimeError("boom")

    pipeline = AgentPipeline(middlewares=[middleware], invoke_fn=core_invoke)
    ctx = AgentContext(
        messages=[Message(role="user", content="hi")],
        session_id="session-1",
        user_id=None,
        config=build_invoke_config("session-1", None, "test-agent"),
        agent_name="test-agent",
        metadata={"query": "hi"},
    )

    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.run(ctx)

    trace = ctx.metadata.get("trace")
    assert trace is not None
    trace.update.assert_called()
    assert ctx.metadata.get("trace_id") == "trace-123"
    tracer.flush.assert_called()
