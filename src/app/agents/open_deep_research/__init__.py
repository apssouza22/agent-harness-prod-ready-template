from src.app.agents.open_deep_research.agent_deep_research import DeepResearchAgent
from src.app.agents.open_deep_research.factory import make_deep_research_agent
from src.app.core.checkpoint.checkpointer import get_checkpointer


async def get_deep_research_agent() -> DeepResearchAgent:
    """Backward-compatible factory wrapper."""
    return await make_deep_research_agent(await get_checkpointer())
