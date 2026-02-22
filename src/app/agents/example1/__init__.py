"""This file contains the prompts for the agent."""


from src.app.agents.example1.agent_example import AgentExample1
from src.app.agents.tools import tools
from src.app.core.checkpoint.checkpointer import get_checkpointer
from src.app.core.llm.llm import LLMService


async def get_agent_example() -> AgentExample1:
    agent = AgentExample1("Agent Example", LLMService("default"), tools, await get_checkpointer())
    await agent.compile()
    return agent
