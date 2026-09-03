"""Core types for the agent middleware pattern.

Defines the context object passed through the middleware chain,
the result type, and the abstract base class that all middlewares extend.

Lifecycle hooks (executed in registration order unless noted):

    before_invoke  →  [graph execution]  →  after_invoke (reverse order)
                        ├─ before_model_call → LLM → after_model_call (reverse)
                        └─ before_tool_call  → tool → after_tool_call  (reverse)
    on_error – called when an exception propagates out of the graph
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from src.app.core.common.config import settings
from src.app.core.common.model.message import Message
from src.app.init import langfuse_callback_handler

InvokeResult = list[Message]

NextFn = Callable[["AgentContext"], Awaitable[InvokeResult]]


@dataclass
class AgentContext:
    """Shared context threaded through the middleware chain.

    Middlewares can read/write ``metadata`` to pass data downstream
    (e.g. ``long_term_memory`` populated by the memory middleware).
    """

    messages: list[Message]
    session_id: str
    user_id: Optional[int]
    config: dict
    agent_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMiddleware(ABC):
    """Base class every middleware extends.

    Subclasses override only the hooks they care about; all hooks
    default to no-ops so a middleware can focus on a single concern.
    """

    # -- invoke-level hooks ---------------------------------------------------

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        """Called before the agent graph is invoked.

        Return ``None`` to continue the chain, or return an
        ``InvokeResult`` to short-circuit (skip the graph entirely).
        """
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        """Called after the agent graph has returned.

        May inspect or transform the result before it reaches the caller.
        """
        return result

    # -- model-call hooks -----------------------------------------------------

    async def before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        """Called before each LLM invocation inside a graph node.

        Return (possibly modified) messages to pass to the model.
        """
        return messages

    async def after_model_call(
        self,
        ctx: AgentContext,
        *,
        response: Any,
        model_name: str,
    ) -> Any:
        """Called after each LLM invocation inside a graph node.

        Return (possibly modified) response.
        """
        return response

    # -- tool-call hooks ------------------------------------------------------

    async def before_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        """Called before a tool is executed inside a graph node.

        Return (possibly modified) tool_args.
        """
        return tool_args

    async def after_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_result: Any,
    ) -> Any:
        """Called after a tool has returned inside a graph node.

        Return (possibly modified) tool_result.
        """
        return tool_result

    # -- error hook -----------------------------------------------------------

    async def on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        """Called when an exception is raised during invocation.

        Return ``None`` to let the next middleware (or the pipeline)
        handle the error.  Return an ``InvokeResult`` to swallow the
        exception and use that result instead.
        """
        return None


def build_invoke_config(
    session_id: str,
    user_id: Optional[int] = None,
    agent_name: str = "",
) -> dict:
    """Build a LangGraph invoke config shared across all agents."""
    metadata: dict[str, str] = {
        "langfuse_session_id": session_id,
        "environment": settings.ENVIRONMENT.value,
        "debug": str(settings.DEBUG).lower(),
    }
    if user_id is not None:
        metadata["langfuse_user_id"] = str(user_id)

    return {
        "callbacks": [langfuse_callback_handler],
        "run_name": agent_name,
        "configurable": {"thread_id": session_id},
        "metadata": metadata,
    }
