from src.app.agents.open_deep_research.agent_deep_research import DeepResearchAgent
from src.app.agents.open_deep_research.factory import make_deep_research_agent
from src.app.core.checkpoint.factory import make_checkpoint_service
from src.app.core.common.config import settings
from src.app.core.db.connection_pool import get_connection_pool
from src.app.core.memory.factory import make_memory_service


async def get_deep_research_agent() -> DeepResearchAgent:
    """Backward-compatible factory wrapper."""
    connection_pool = await get_connection_pool()
    checkpoint_service = make_checkpoint_service(connection_pool=connection_pool)
    memory_service = make_memory_service(settings)
    return await make_deep_research_agent(
        await checkpoint_service.get_checkpointer(),
        memory_service=memory_service,
    )
