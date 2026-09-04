"""Chatbot agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.tools import tools
from src.app.core.langfuse.client import LangfuseTracer


async def make_chatbot_agent(
    checkpointer: AsyncPostgresSaver | None,
    langfuse_tracer: LangfuseTracer | None = None,
) -> AgentChatbot:
    """Create and compile a chatbot agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.

    Returns:
        AgentChatbot: Compiled chatbot agent instance.
    """
    agent = AgentChatbot("Chatbot", tools, checkpointer, langfuse_tracer=langfuse_tracer)
    await agent.compile()
    return agent
