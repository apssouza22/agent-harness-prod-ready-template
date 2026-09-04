"""Chatbot agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.chatbot.agent_chatbot import (
    AgentChatbot,
    build_chatbot_trace_metadata,
    build_chatbot_trace_output,
    chatbot_model,
)
from src.app.agents.tools import tools
from src.app.core.common.config import settings
from src.app.core.langfuse import LangfuseTracer, LangfuseTracingMiddleware
from src.app.core.context import SummarizationMiddleware, TrimLongMessagesMiddleware
from src.app.core.guardrails import GuardrailMiddleware
from src.app.core.memory import MemoryMiddleware
from src.app.core.metrics import LlmMetricsMiddleware
from src.app.core.middleware import ErrorHandlingMiddleware, LoggingMiddleware


async def make_chatbot_agent(
    checkpointer: AsyncPostgresSaver | None,
    langfuse_tracer: LangfuseTracer | None = None,
) -> AgentChatbot:
    """Create and compile a chatbot agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.
        langfuse_tracer: Optional Langfuse tracer for observability.

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
        SummarizationMiddleware(
            llm=chatbot_model,
            model_name=f"openai:{settings.DEFAULT_LLM_MODEL}",
        ),
        TrimLongMessagesMiddleware(
            llm=chatbot_model,
            max_tokens=settings.MAX_TOKENS,
        ),
    ]
    agent = AgentChatbot("Chatbot", tools, checkpointer, middlewares=middlewares)
    await agent.compile()
    return agent
