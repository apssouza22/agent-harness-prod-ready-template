import asyncio
import os
from typing import Any, Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.app.core.agentic.agent_base import AgentAbstract
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.model.message import Message
from src.app.core.llm.llm import LLMService


class TextSQLDeepAgent(AgentAbstract):
    """SQL Deep Agent that can interact with a SQL database using natural language instructions."""

    def __init__(self, name: str, checkpointer: AsyncPostgresSaver = None):
        super().__init__(name, LLMService("text_sql"), [], checkpointer)
        self.agent = create_sql_deep_agent()

    async def agent_invoke(
        self,
        agent_input: dict[str, Any],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        """Invoke the SQL Deep Agent with the given input and return its response."""
        config = await self._create_config(session_id, user_id)

        response =  await self.agent.ainvoke({
            "messages": [{"role": "user", "content": agent_input.get("query", "")}]
        }, config=config)
        return process_messages(response["messages"])


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    model = ChatOpenAI(model="gpt-5-mini", reasoning={"effort": "medium"}, temperature=0)

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    agent = create_deep_agent(
        model=model,
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/"
        ],  # Specialized workflows (query-writing, schema-exploration)
        tools=sql_tools,  # SQL database tools
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir),  # Persistent file storage
    )

    return agent
