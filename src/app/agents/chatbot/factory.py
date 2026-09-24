"""Chatbot agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.chatbot.agent_chatbot import (
    AgentChatbot,
    build_chatbot_trace_metadata,
    build_chatbot_trace_output,
    chatbot_model,
)
from src.app.core.tools import tools
from src.app.core.common.config import settings
from src.app.core.langfuse import LangfuseTracer, LangfuseTracingMiddleware
from src.app.core.mcp.manager import McpManager
from src.app.core.context import SummarizationMiddleware, TrimLongMessagesMiddleware
from src.app.core.guardrails import GuardrailMiddleware
from src.app.core.dialogue_state import DialogueStateMiddleware
from src.app.core.memory import MemoryMiddleware
from src.app.core.metrics import LlmMetricsMiddleware
from src.app.core.llm.factory import resolve_model_identifier
from src.app.core.middleware import ErrorHandlingMiddleware, LoggingMiddleware


async def make_chatbot_agent(
    checkpointer: AsyncPostgresSaver | None,
    langfuse_tracer: LangfuseTracer | None = None,
    mcp_manager: McpManager | None = None,
) -> AgentChatbot:
    """Create and compile a chatbot agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.
        langfuse_tracer: Optional Langfuse tracer for observability.
        mcp_manager: Optional MCP manager for external tool discovery and calls.

    Returns:
        AgentChatbot: Compiled chatbot agent instance.
    """
    middlewares = [
        LangfuseTracingMiddleware(
            langfuse_tracer=langfuse_tracer,
            trace_name="chatbot_request",
            environment=settings.ENVIRONMENT.value,
            build_trace_metadata=build_chatbot_trace_metadata,
            build_trace_output=build_chatbot_trace_output,
        ),
        LoggingMiddleware(),
        GuardrailMiddleware(langfuse_tracer=langfuse_tracer),
        LlmMetricsMiddleware(),
        ErrorHandlingMiddleware(),
        MemoryMiddleware(),
        DialogueStateMiddleware(),
        SummarizationMiddleware(
            llm=chatbot_model,
            model_name=resolve_model_identifier(),
        ),
        TrimLongMessagesMiddleware(
            max_tokens=settings.MAX_TOKENS,
        ),
    ]
    agent = AgentChatbot(
        "Chatbot",
        tools,
        checkpointer,
        middlewares=middlewares,
        mcp_manager=mcp_manager,
    )
    await agent.compile()
    return agent
