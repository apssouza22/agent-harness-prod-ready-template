"""LangGraph node error handlers for graceful recovery after retry exhaustion."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import NodeError
from langgraph.types import Command

from src.app.core.common.logging import logger
from src.app.core.llm.llm_utils import record_llm_error
from src.app.core.metrics.metrics import graph_node_failures_total

LLM_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching the language model right now. Please try again in a moment."
)
DEEP_RESEARCH_FAILURE_MESSAGE = (
    "Research could not be completed due to a temporary service issue. Please try again shortly."
)


def _record_node_failure(agent_name: str, error: NodeError) -> None:
    graph_node_failures_total.labels(agent_name=agent_name, node=error.node).inc()
    logger.exception(
        "graph_node_failed_after_retries",
        agent_name=agent_name,
        node=error.node,
        error_type=type(error.error).__name__,
        error=str(error.error),
    )


def create_chat_node_error_handler(
    *,
    agent_name: str,
    model_name: str,
    fallback_goto: str,
) -> Callable[[Any, NodeError], Command]:
    """Return an error handler that routes to a fallback node with an apology message."""

    def chat_node_error_handler(state: Any, error: NodeError) -> Command:
        _record_node_failure(agent_name, error)
        record_llm_error(model_name, agent_name)
        return Command(
            update={
                "messages": [AIMessage(content=LLM_UNAVAILABLE_MESSAGE)],
                "last_node_error": str(error.error),
                "failed_node": error.node,
            },
            goto=fallback_goto,
        )

    return chat_node_error_handler


def create_tool_node_error_handler(
    *,
    agent_name: str,
    fallback_goto: str,
) -> Callable[[Any, NodeError], Command]:
    """Return an error handler that injects tool error messages and continues the graph."""

    def tool_node_error_handler(state: Any, error: NodeError) -> Command:
        _record_node_failure(agent_name, error)
        tool_messages: list[ToolMessage] = []
        messages = state.messages if hasattr(state, "messages") else state.get("messages", [])
        if messages:
            last_message = messages[-1]
            tool_calls = getattr(last_message, "tool_calls", None) or []
            for tool_call in tool_calls:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool execution failed after retries: {error.error}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

        return Command(
            update={
                "messages": tool_messages,
                "last_node_error": str(error.error),
                "failed_node": error.node,
            },
            goto=fallback_goto,
        )

    return tool_node_error_handler


def create_deep_research_error_handler(
    *,
    agent_name: str,
    model_name: str,
    fallback_goto: str,
) -> Callable[[Any, NodeError], Command]:
    """Return an error handler for deep-research LLM nodes."""

    def deep_research_error_handler(state: Any, error: NodeError) -> Command:
        _record_node_failure(agent_name, error)
        record_llm_error(model_name, agent_name)
        return Command(
            update={
                "messages": [AIMessage(content=DEEP_RESEARCH_FAILURE_MESSAGE)],
                "last_node_error": str(error.error),
                "failed_node": error.node,
            },
            goto=fallback_goto,
        )

    return deep_research_error_handler
