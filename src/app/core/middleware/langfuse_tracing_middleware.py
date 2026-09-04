"""Langfuse tracing middleware for per-request observation lifecycle."""

import time
from contextlib import ExitStack
from typing import Any, Callable, Optional

from src.app.core.langfuse.client import LangfuseTracer
from src.app.core.common.logging import logger
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult


def _default_build_trace_input(ctx: AgentContext) -> dict[str, Any]:
    query = ctx.metadata.get("query")
    if query:
        return {"query": query}
    if ctx.messages:
        return {"query": ctx.messages[-1].content}
    return {}


def _default_build_trace_metadata(ctx: AgentContext) -> dict[str, Any]:
    return dict(ctx.metadata.get("trace_metadata", {}))


def _default_build_trace_output(ctx: AgentContext, result: InvokeResult) -> dict[str, Any]:
    execution_time = time.time() - ctx.metadata.get("_trace_start_time", time.time())
    answer = result[-1].content if result else ""
    return {
        "answer": answer,
        "execution_time": execution_time,
    }


class LangfuseTracingMiddleware(AgentMiddleware):
    """Creates root Langfuse observations and wires LangGraph callbacks per request."""

    def __init__(
        self,
        langfuse_tracer: Optional[LangfuseTracer],
        trace_name: str,
        *,
        environment: Optional[str] = None,
        build_trace_input: Optional[Callable[[AgentContext], dict]] = None,
        build_trace_metadata: Optional[Callable[[AgentContext], dict]] = None,
        build_trace_output: Optional[Callable[[AgentContext, InvokeResult], dict]] = None,
    ) -> None:
        self._langfuse_tracer = langfuse_tracer
        self._trace_name = trace_name
        self._environment = environment
        self._build_trace_input = build_trace_input or _default_build_trace_input
        self._build_trace_metadata = build_trace_metadata or _default_build_trace_metadata
        self._build_trace_output = build_trace_output or _default_build_trace_output

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        if not self._langfuse_tracer or not self._langfuse_tracer.client:
            return None

        ctx.metadata["_trace_start_time"] = time.time()
        exit_stack = ExitStack()
        ctx.metadata["_langfuse_exit_stack"] = exit_stack

        try:
            trace_input = self._build_trace_input(ctx)
            trace_metadata = self._build_trace_metadata(ctx)
            user_id = ctx.metadata.get("user_id", ctx.user_id)
            tags = ctx.metadata.get("trace_tags")

            observation = exit_stack.enter_context(
                self._langfuse_tracer.trace_agent_request(
                    self._trace_name,
                    input_data=trace_input,
                    user_id=user_id,
                    session_id=ctx.session_id,
                    metadata=trace_metadata,
                    tags=tags,
                    environment=self._environment,
                )
            )
            ctx.metadata["trace"] = observation
            ctx.metadata["langfuse_tracer"] = self._langfuse_tracer

            callback_handler = self._langfuse_tracer.get_callback_handler()
            if callback_handler is not None:
                callbacks = ctx.config.setdefault("callbacks", [])
                if callback_handler not in callbacks:
                    callbacks.append(callback_handler)

                graph_config = ctx.config.setdefault("graph_config", {})
                graph_callbacks = graph_config.setdefault("callbacks", [])
                if callback_handler not in graph_callbacks:
                    graph_callbacks.append(callback_handler)
        except Exception:
            logger.warning("langfuse_tracing_middleware_before_invoke_failed", exc_info=True)
            exit_stack.close()
            ctx.metadata.pop("_langfuse_exit_stack", None)
            ctx.metadata.pop("trace", None)

        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        trace = ctx.metadata.get("trace")
        exit_stack = ctx.metadata.get("_langfuse_exit_stack")

        if trace is not None:
            try:
                output = self._build_trace_output(ctx, result)
                trace.update(output=output)
            except Exception:
                logger.warning("langfuse_tracing_middleware_after_invoke_update_failed", exc_info=True)

        if self._langfuse_tracer:
            trace_id = self._langfuse_tracer.get_trace_id(trace)
            if trace_id:
                ctx.metadata["trace_id"] = trace_id
            self._langfuse_tracer.flush()

        if exit_stack is not None:
            try:
                exit_stack.close()
            except Exception:
                logger.warning("langfuse_tracing_middleware_exit_stack_close_failed", exc_info=True)
            ctx.metadata.pop("_langfuse_exit_stack", None)

        return result

    async def on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        trace = ctx.metadata.get("trace")
        exit_stack = ctx.metadata.get("_langfuse_exit_stack")

        if trace is not None:
            try:
                trace.update(output={"error": str(error)}, level="ERROR")
            except Exception:
                logger.warning("langfuse_tracing_middleware_on_error_update_failed", exc_info=True)

        if self._langfuse_tracer:
            trace_id = self._langfuse_tracer.get_trace_id(trace)
            if trace_id:
                ctx.metadata["trace_id"] = trace_id
            self._langfuse_tracer.flush()

        if exit_stack is not None:
            try:
                exit_stack.close()
            except Exception:
                logger.warning("langfuse_tracing_middleware_exit_stack_close_failed", exc_info=True)
            ctx.metadata.pop("_langfuse_exit_stack", None)

        return None
