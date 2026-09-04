"""Deep research agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.open_deep_research.agent_deep_research import DeepResearchAgent


async def make_deep_research_agent(checkpointer: AsyncPostgresSaver | None) -> DeepResearchAgent:
    """Create and compile a deep research agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.

    Returns:
        DeepResearchAgent: Compiled deep research agent instance.
    """
    agent = DeepResearchAgent("Deep Research", checkpointer)
    await agent.compile()
    return agent
