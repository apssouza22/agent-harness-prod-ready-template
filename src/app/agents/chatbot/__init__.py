"""This file contains the prompts for the agent."""

from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.chatbot.factory import make_chatbot_agent
from src.app.core.checkpoint.checkpointer import get_checkpointer


async def get_agent_example() -> AgentChatbot:
    """Backward-compatible factory wrapper."""
    return await make_chatbot_agent(await get_checkpointer())
