"""This file contains the prompts for the agent."""

from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.chatbot.factory import make_chatbot_agent
from src.app.core.checkpoint.factory import make_checkpoint_service
from src.app.core.db.connection_pool import get_connection_pool


async def get_agent_example() -> AgentChatbot:
    """Backward-compatible factory wrapper."""
    connection_pool = await get_connection_pool()
    checkpoint_service = make_checkpoint_service(connection_pool=connection_pool)
    return await make_chatbot_agent(await checkpoint_service.get_checkpointer())
