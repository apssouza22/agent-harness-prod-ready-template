"""This file contains the prompts for the agent."""

from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.chatbot.factory import make_chatbot_agent
from src.app.core.checkpoint.factory import make_checkpoint_service
from src.app.core.common.config import settings
from src.app.core.db.connection_pool import get_connection_pool
from src.app.core.dialogue_state.factory import make_dialogue_state_service
from src.app.core.memory.factory import make_memory_service


async def get_agent_example() -> AgentChatbot:
    """Backward-compatible factory wrapper."""
    connection_pool = await get_connection_pool()
    checkpoint_service = make_checkpoint_service(connection_pool=connection_pool)
    memory_service = make_memory_service(settings)
    dialogue_state_service = make_dialogue_state_service(settings)
    return await make_chatbot_agent(
        await checkpoint_service.get_checkpointer(),
        memory_service=memory_service,
        dialogue_state_service=dialogue_state_service,
    )
