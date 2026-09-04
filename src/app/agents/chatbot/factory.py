"""Chatbot agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.tools import tools


async def make_chatbot_agent(checkpointer: AsyncPostgresSaver | None) -> AgentChatbot:
    """Create and compile a chatbot agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.

    Returns:
        AgentChatbot: Compiled chatbot agent instance.
    """
    agent = AgentChatbot("Chatbot", tools, checkpointer)
    await agent.compile()
    return agent
