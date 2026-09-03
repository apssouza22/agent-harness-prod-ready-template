"""Compiled graph ready for synchronous and asynchronous execution."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from langgraph.typing import ContextT, InputT, OutputT, StateT

from src.app.core.graph.types import Command


class StateGraphCompiled:
    """Compiled workflow ready to run."""

    def __init__(
        self,
        state_graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
    ) -> None:
        self._state_graph = state_graph

    @property
    def state_graph(self) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        """Underlying LangGraph compiled graph (for advanced use)."""
        return self._state_graph

    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | Any | None = None,
    ):
        """Run the compiled workflow synchronously."""
        return self._state_graph.invoke(input, config, context=context)

    async def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | Any | None = None,
    ):
        """Run the compiled workflow asynchronously."""
        return await self._state_graph.ainvoke(input, config, context=context)

    def stream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream the compiled workflow synchronously."""
        yield from self._state_graph.stream(input, config, **kwargs)

    def astream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream the compiled workflow asynchronously."""
        return self._state_graph.astream(input, config, **kwargs)

    def get_state(self, config: RunnableConfig, **kwargs: Any) -> StateSnapshot:
        """Return the latest graph state for a thread."""
        return self._state_graph.get_state(config, **kwargs)

    def get_graph(self):
        """Return the graph structure for visualization or introspection."""
        return self._state_graph.get_graph()
