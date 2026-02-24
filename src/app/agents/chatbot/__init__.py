"""This file contains the prompts for the agent."""


from src.app.agents.chatbot.agent_chatbot import AgentChatbot
from src.app.agents.tools import tools
from src.app.core.checkpoint.checkpointer import get_checkpointer


async def get_agent_example() -> AgentChatbot:
    agent = AgentChatbot("Agent Example", tools, await get_checkpointer())
    await agent.compile()
    return agent
