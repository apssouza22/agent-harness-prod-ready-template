"""Text-to-SQL agent factory."""

from src.app.agents.text_to_sql.text_sql_agent import TextSQLDeepAgent


async def make_text_to_sql_agent() -> TextSQLDeepAgent:
    """Create a text-to-SQL deep agent.

    Returns:
        TextSQLDeepAgent: Text-to-SQL agent instance.
    """
    return TextSQLDeepAgent(name="Text-to-SQL")
