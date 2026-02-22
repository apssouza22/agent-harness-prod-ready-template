"""Factory for the Text-to-SQL Deep Agent.

Uses importlib to load from the 'text-to-sql' directory since hyphens
are not valid in Python package names.
"""

import importlib.util
from pathlib import Path

_AGENT_MODULE_PATH = Path(__file__).parent / "text-to-sql" / "text_sql_agent.py"
_spec = importlib.util.spec_from_file_location("text_sql_agent", _AGENT_MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

TextSQLDeepAgent = _module.TextSQLDeepAgent

_agent_instance: TextSQLDeepAgent | None = None


async def get_text_sql_agent() -> TextSQLDeepAgent:
    """Factory to create a TextSQLDeepAgent singleton instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TextSQLDeepAgent(name="Text-to-SQL")
    return _agent_instance
