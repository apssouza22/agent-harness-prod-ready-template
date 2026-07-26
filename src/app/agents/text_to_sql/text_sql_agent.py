import os
from typing import Any, Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import PIIMiddleware
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from src.app.core.llm.factory import create_openai_chat_model
from src.app.core.middleware import (
    AgentContext,
    AgentPipeline,
    build_invoke_config,
    ErrorHandlingMiddleware,
    GuardrailMiddleware,
    LoggingMiddleware,
)
from src.app.core.common.config import settings
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.model.message import Message


class TextSQLDeepAgent:
    """SQL Deep Agent that can interact with a SQL database using natural language instructions."""

    def __init__(self, name: str):
        self.name = name
        self.agent = create_sql_deep_agent()
        self._pipeline = AgentPipeline(
            middlewares=[
                LoggingMiddleware(),
                ErrorHandlingMiddleware(),
                GuardrailMiddleware(),
            ],
            invoke_fn=self._core_invoke,
        )

    async def agent_invoke(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        """Invoke the SQL Deep Agent through the middleware pipeline."""
        ctx = AgentContext(
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            config=build_invoke_config(session_id, user_id, self.name),
            agent_name=self.name,
            metadata={"model_name": "gpt-5-mini"},
        )
        return await self._pipeline.run(ctx)

    async def _core_invoke(self, ctx: AgentContext) -> list[Message]:
        """Core agent invocation without cross-cutting concerns."""
        query = ctx.messages[-1].content if ctx.messages else ""

        response = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=ctx.config,
        )
        return process_messages(response["messages"])


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    model = create_openai_chat_model(model="gpt-5-mini", reasoning={"effort": "medium"}, temperature=0)

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    agent = create_deep_agent(
        model=model,
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/"
        ],  # Specialized workflows (query-writing, schema-exploration)
        middleware=[PIIMiddleware("email")],
        tools=sql_tools,  # SQL database tools
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir),  # Persistent file storage
    )

    return agent
