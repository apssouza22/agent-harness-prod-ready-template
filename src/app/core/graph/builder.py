"""Graph builder that wraps LangGraph behind a single domain API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Self, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.typing import StateT

from src.app.core.graph.compiled import StateGraphCompiled
from src.app.core.graph.config import FaultToleranceConfig
from src.app.core.graph.policies import build_llm_timeout, build_retry_policy
from src.app.core.graph.types import Checkpointer, Command, END, MessagesState, NodeError, ToolNode
from src.app.core.middleware.pipeline import MiddlewareManager, middleware_tool_wrappers

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build and compile agent workflows behind a single domain API."""

    def __init__(
        self,
        state_schema: type[StateT] | type[TypedDict],
        context_schema: type[Any] | None = None,
        *,
        input_schema: type[Any] | None = None,
        output_schema: type[Any] | None = None,
    ) -> None:
        self._middleware_manager = None
        self._graph = StateGraph(
            state_schema,
            context_schema,
            input_schema=input_schema,
            output_schema=output_schema,
        )
        ft = FaultToleranceConfig()
        self.set_node_defaults(
            retry_policy=build_retry_policy(ft),
            timeout=build_llm_timeout(ft),
        )

    def set_middleware_manager(self, manager: MiddlewareManager) -> Self:
        self._middleware_manager = manager
        return self

    def set_node_defaults(self, **kwargs: Any) -> Self:
        self._graph.set_node_defaults(**kwargs)
        return self

    def add_node(self, node: str, action: Any | None = None, /, **kwargs: Any) -> Self:
        if isinstance(action, StateGraphCompiled):
            action = action.state_graph
        if isinstance(action, ToolNode):
            action = self._apply_tool_middleware(action)
        self._graph.add_node(node, action, **kwargs)
        return self

    def set_entry_point(self, node: str) -> Self:
        self._graph.set_entry_point(node)
        return self

    def add_tool_node(
        self,
        node: str,
        tools: Sequence[Any],
        /,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Add a ``ToolNode`` wired to the configured middleware manager."""
        tool_node = ToolNode(
            tools,
            name=name or node,
            **middleware_tool_wrappers(self._middleware_manager),
        )
        return self.add_node(node, tool_node, **kwargs)

    def add_edge(self, start_key: str | Any, end_key: str | Any) -> Self:
        self._graph.add_edge(start_key, end_key)
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Hashable | Sequence[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None = None,
    ) -> Self:
        self._graph.add_conditional_edges(source, path, path_map)
        return self

    def compile(
        self,
        *,
        name: str | None = None,
        checkpointer: Checkpointer = None,
        **kwargs: Any,
    ) -> StateGraphCompiled:
        return StateGraphCompiled(self._graph.compile(name=name, checkpointer=checkpointer, **kwargs))

    def _apply_tool_middleware(self, tool_node: ToolNode) -> ToolNode:
        wrappers = middleware_tool_wrappers(self._middleware_manager)
        if not wrappers:
            return tool_node

        if tool_node._wrap_tool_call is None:
            tool_node._wrap_tool_call = wrappers["wrap_tool_call"]
        if tool_node._awrap_tool_call is None:
            tool_node._awrap_tool_call = wrappers["awrap_tool_call"]
        return tool_node


async def default_error_handler(state: MessagesState, error: NodeError) -> Command:
    """Return a user-facing message when the text-to-SQL workflow cannot complete."""
    logger.error("Graph node '%s' failed with error: %s", error.node, error.error)

    ai_message = AIMessage(content="I was unable to proceed with the request.")
    return Command(
        goto=END,
        update={"messages": [ai_message]},
    )
