"""Unit tests for LangGraph fault tolerance helpers."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import NodeError
from langgraph.types import RetryPolicy, TimeoutPolicy

from src.app.core.fault_tolerance.error_handlers import (
    LLM_UNAVAILABLE_MESSAGE,
    create_chat_node_error_handler,
    create_tool_node_error_handler,
)
from src.app.core.fault_tolerance.policies import (
    get_llm_retry_policy,
    get_llm_timeout_policy,
    get_tool_retry_policy,
)
from src.app.core.graph import END
from src.app.core.common.config import settings
from src.app.core.common.model.graph import GraphState


def test_llm_retry_policy_uses_settings():
    policy = get_llm_retry_policy()
    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts == settings.GRAPH_LLM_RETRY_MAX_ATTEMPTS


def test_tool_retry_policy_uses_settings():
    policy = get_tool_retry_policy()
    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts == settings.GRAPH_TOOL_RETRY_MAX_ATTEMPTS


def test_llm_timeout_policy_uses_settings():
    policy = get_llm_timeout_policy()
    assert isinstance(policy, TimeoutPolicy)
    assert policy.run_timeout == settings.GRAPH_LLM_RUN_TIMEOUT
    assert policy.idle_timeout == settings.GRAPH_LLM_IDLE_TIMEOUT


@pytest.mark.asyncio
async def test_chat_error_handler_routes_to_fallback():
    handler = create_chat_node_error_handler(
        agent_name="test-agent",
        model_name="gpt-test",
        fallback_goto=END,
    )
    state = GraphState(messages=[HumanMessage(content="hello")])
    command = await handler(state, NodeError(node="chat", error=RuntimeError("provider down")))

    assert command.goto == END
    assert command.update["failed_node"] == "chat"
    assert command.update["messages"][0].content == LLM_UNAVAILABLE_MESSAGE


@pytest.mark.asyncio
async def test_tool_error_handler_returns_tool_messages():
    handler = create_tool_node_error_handler(agent_name="test-agent", fallback_goto="chat")
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1", "type": "tool_call"}],
    )
    state = GraphState(messages=[HumanMessage(content="hello"), ai_message])
    command = await handler(state, NodeError(node="tool_call", error=ConnectionError("reset")))

    assert command.goto == "chat"
    assert len(command.update["messages"]) == 1
    assert command.update["messages"][0].name == "search"
    assert "failed after retries" in command.update["messages"][0].content
