"""Factory for the Text-to-SQL Deep Agent."""

from src.app.agents.text_to_sql.text_sql_agent import TextSQLDeepAgent

_agent_instance: TextSQLDeepAgent | None = None


async def get_text_sql_agent() -> TextSQLDeepAgent:
    """Factory to create a TextSQLDeepAgent singleton instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TextSQLDeepAgent(name="Text-to-SQL")
    return _agent_instance
