"""Factory for the Text-to-SQL Deep Agent."""

from src.app.agents.text_to_sql.factory import make_text_to_sql_agent
from src.app.agents.text_to_sql.text_sql_agent import TextSQLDeepAgent

_agent_instance: TextSQLDeepAgent | None = None


async def get_text_sql_agent() -> TextSQLDeepAgent:
    """Backward-compatible singleton factory wrapper."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = await make_text_to_sql_agent()
    return _agent_instance
