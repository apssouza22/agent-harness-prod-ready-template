"""Deep research agent factory."""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.agents.open_deep_research.agent_deep_research import DeepResearchAgent
from src.app.core.langfuse.client import LangfuseTracer


async def make_deep_research_agent(
    checkpointer: AsyncPostgresSaver | None,
    langfuse_tracer: LangfuseTracer | None = None,
) -> DeepResearchAgent:
    """Create and compile a deep research agent.

    Args:
        checkpointer: LangGraph async Postgres checkpointer, or None.

    Returns:
        DeepResearchAgent: Compiled deep research agent instance.
    """
    agent = DeepResearchAgent("Deep Research", checkpointer, langfuse_tracer=langfuse_tracer)
    await agent.compile()
    return agent
