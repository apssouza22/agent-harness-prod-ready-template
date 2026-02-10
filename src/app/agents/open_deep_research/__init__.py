from src.app.agents.open_deep_research.agent_deep_research import DeepResearchAgent
from src.app.core.checkpoint.checkpointer import get_checkpointer
from src.app.core.llm.llm import llm_service


async def get_deep_research_agent() -> DeepResearchAgent:
    """Factory to create and compile a DeepResearchAgent instance."""
    agent = DeepResearchAgent("Deep Research", llm_service, await get_checkpointer())
    await agent.compile()
    return agent
