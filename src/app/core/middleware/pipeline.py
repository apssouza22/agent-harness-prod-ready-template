"""Agent pipeline that composes class-based middlewares with lifecycle hooks.

The ``MiddlewareManager`` dispatches calls to every registered middleware
for each lifecycle hook.  ``AgentPipeline`` drives the top-level
``before_invoke → invoke → after_invoke`` flow and exposes the manager
so that graph nodes can trigger ``before/after_model_call`` and
``before/after_tool_call`` hooks.
"""

from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Optional, Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult, NextFn

_active_middleware_manager: ContextVar[Optional["MiddlewareManager"]] = ContextVar(
    "_active_middleware_manager",
    default=None,
)


def get_active_middleware_manager() -> Optional["MiddlewareManager"]:
    """Return the middleware manager for the current agent invocation."""
    return _active_middleware_manager.get()


def middleware_tool_wrappers(
    manager: Optional["MiddlewareManager"] = None,
) -> dict[str, Any]:
    """Return LangGraph ToolNode wrappers that run middleware tool hooks."""
    active_manager = manager or get_active_middleware_manager()
    if active_manager is None:
        return {}

    def _tool_name(request: ToolCallRequest) -> str:
        return request.tool_call["name"]

    def _tool_args(request: ToolCallRequest) -> dict:
        return dict(request.tool_call.get("args") or {})

    def _with_tool_args(request: ToolCallRequest, tool_args: dict) -> ToolCallRequest:
        tool_call = dict(request.tool_call)
        tool_call["args"] = tool_args
        return replace(request, tool_call=tool_call)

    def wrap_tool_call(request: ToolCallRequest, handler):
        return handler(request)

    async def awrap_tool_call(request: ToolCallRequest, handler):
        manager = active_manager
        ctx = manager.active_ctx
        if manager and ctx:
            tool_args = await manager.run_before_tool_call(
                ctx,
                tool_name=_tool_name(request),
                tool_args=_tool_args(request),
            )
            request = _with_tool_args(request, tool_args)

        result = await handler(request)

        if manager and ctx:
            return await manager.run_after_tool_call(
                ctx,
                tool_name=_tool_name(request),
                tool_result=result,
            )
        return result

    return {
        "wrap_tool_call": wrap_tool_call,
        "awrap_tool_call": awrap_tool_call,
    }


class MiddlewareManager:
    """Dispatches lifecycle hooks to all registered middlewares.

    The manager is stored on the pipeline and made available to graph
    nodes via ``pipeline.manager`` so they can call model/tool hooks.
    """

    def __init__(self, middlewares: Sequence[AgentMiddleware]) -> None:
        self._middlewares = list(middlewares)
        self._active_ctx: Optional[AgentContext] = None

    @property
    def active_ctx(self) -> Optional[AgentContext]:
        """The AgentContext for the currently running invocation."""
        return self._active_ctx

    def set_active_ctx(self, ctx: Optional[AgentContext]) -> None:
        self._active_ctx = ctx

    # -- invoke-level hooks ---------------------------------------------------

    async def run_before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        """Run ``before_invoke`` on each middleware; short-circuit on first non-None."""
        for mw in self._middlewares:
            result = await mw.before_invoke(ctx)
            if result is not None:
                return result
        return None

    async def run_after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        """Run ``after_invoke`` on each middleware in reverse (stack-unwind) order."""
        for mw in reversed(self._middlewares):
            result = await mw.after_invoke(ctx, result)
        return result

    async def run_on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        """Run ``on_error`` on each middleware; first non-None result wins."""
        for mw in self._middlewares:
            result = await mw.on_error(ctx, error)
            if result is not None:
                return result
        return None

    # -- model-call hooks -----------------------------------------------------

    async def run_before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        for mw in self._middlewares:
            messages = await mw.before_model_call(ctx, messages=messages, model_name=model_name)
        return messages

    async def run_after_model_call(
        self,
        ctx: AgentContext,
        *,
        response: Any,
        model_name: str,
    ) -> Any:
        for mw in reversed(self._middlewares):
            response = await mw.after_model_call(ctx, response=response, model_name=model_name)
        return response

    # -- tool-call hooks ------------------------------------------------------

    async def run_before_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        for mw in self._middlewares:
            tool_args = await mw.before_tool_call(ctx, tool_name=tool_name, tool_args=tool_args)
        return tool_args

    async def run_after_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_result: Any,
    ) -> Any:
        for mw in reversed(self._middlewares):
            tool_result = await mw.after_tool_call(ctx, tool_name=tool_name, tool_result=tool_result)
        return tool_result


async def invoke_model(
    model,
    model_input: LanguageModelInput,
    model_name: str,
    *,
    config: RunnableConfig | None = None,
    manager: Optional["MiddlewareManager"] = None,
    ctx: Optional[AgentContext] = None,
    run_before_hooks: bool = True,
) -> Any:
    """Invoke an LLM through the active middleware pipeline."""
    active_manager = manager or get_active_middleware_manager()
    active_ctx = ctx or (active_manager.active_ctx if active_manager else None)
    hook_messages = model_input if isinstance(model_input, list) else []

    if active_manager and active_ctx and run_before_hooks:
        hook_messages = await active_manager.run_before_model_call(
            active_ctx,
            messages=hook_messages,
            model_name=model_name,
        )
        if hook_messages is not model_input and isinstance(model_input, list):
            model_input = hook_messages

    response = await model.ainvoke(model_input, config)

    if active_manager and active_ctx:
        response = await active_manager.run_after_model_call(
            active_ctx,
            response=response,
            model_name=model_name,
        )
    return response


class AgentPipeline:
    """Composable middleware pipeline for agent invocations.

    Usage::

        pipeline = AgentPipeline(
            middlewares=[LoggingMiddleware(), ErrorHandlingMiddleware(), MemoryMiddleware()],
            invoke_fn=agent.core_invoke,
        )
        result = await pipeline.run(ctx)
    """

    def __init__(
        self,
        middlewares: Sequence[AgentMiddleware],
        invoke_fn: NextFn,
    ) -> None:
        self.manager = MiddlewareManager(middlewares)
        self._invoke_fn = invoke_fn

    async def run(self, ctx: AgentContext) -> InvokeResult:
        """Execute the full middleware lifecycle around the core invoke function."""
        manager_token = _active_middleware_manager.set(self.manager)
        self.manager.set_active_ctx(ctx)
        try:
            short_circuit = await self.manager.run_before_invoke(ctx)
            if short_circuit is not None:
                return short_circuit

            try:
                result = await self._invoke_fn(ctx)
            except Exception as e:
                error_result = await self.manager.run_on_error(ctx, e)
                if error_result is not None:
                    return error_result
                raise

            return await self.manager.run_after_invoke(ctx, result)
        finally:
            self.manager.set_active_ctx(None)
            _active_middleware_manager.reset(manager_token)
