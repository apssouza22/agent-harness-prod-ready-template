import os
import time
from typing import Any, Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import PIIMiddleware
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from src.app.core.llm.factory import create_openai_chat_model
from src.app.core.langfuse.client import LangfuseTracer
from src.app.core.middleware import (
    AgentContext,
    AgentPipeline,
    build_invoke_config,
    ErrorHandlingMiddleware,
    GuardrailMiddleware,
    LangfuseTracingMiddleware,
    LlmMetricsMiddleware,
    LoggingMiddleware,
)
from src.app.core.common.config import settings
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.model.message import Message


class TextSQLDeepAgent:
    """SQL Deep Agent that can interact with a SQL database using natural language instructions."""

    def __init__(self, name: str, langfuse_tracer: Optional[LangfuseTracer] = None):
        self.name = name
        self.agent = create_sql_deep_agent()
        self._last_trace_id: Optional[str] = None
        self._pipeline = AgentPipeline(
            middlewares=[
                LangfuseTracingMiddleware(
                    langfuse_tracer=langfuse_tracer,
                    trace_name="text_to_sql_request",
                    environment=settings.ENVIRONMENT.value,
                    build_trace_metadata=self._build_trace_metadata,
                    build_trace_output=self._build_trace_output,
                ),
                LoggingMiddleware(),
                LlmMetricsMiddleware(),
                ErrorHandlingMiddleware(),
                GuardrailMiddleware(),
            ],
            invoke_fn=self._core_invoke,
        )

    @property
    def last_trace_id(self) -> Optional[str]:
        return self._last_trace_id

    async def agent_invoke(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        """Invoke the SQL Deep Agent through the middleware pipeline."""
        query = messages[-1].content if messages else ""
        ctx = AgentContext(
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            config=build_invoke_config(session_id, user_id, self.name),
            agent_name=self.name,
            metadata={
                "query": query,
                "user_id": user_id,
                "model_name": "gpt-5-mini",
                "trace_metadata": {
                    "service": "text_to_sql",
                    "model": "gpt-5-mini",
                },
            },
        )
        result = await self._pipeline.run(ctx)
        self._last_trace_id = ctx.metadata.get("trace_id")
        return result

    def _build_trace_metadata(self, ctx: AgentContext) -> dict[str, Any]:
        return dict(ctx.metadata.get("trace_metadata", {}))

    def _build_trace_output(self, ctx: AgentContext, result: list[Message]) -> dict[str, Any]:
        execution_time = time.time() - ctx.metadata.get("_trace_start_time", time.time())
        answer = result[-1].content if result else ""
        return {
            "answer": answer,
            "message_count": len(result),
            "execution_time": execution_time,
        }

    async def _core_invoke(self, ctx: AgentContext) -> list[Message]:
        """Core agent invocation without cross-cutting concerns."""
        query = ctx.messages[-1].content if ctx.messages else ""

        response = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=ctx.config,
        )
        ctx.metadata["graph_result"] = response
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
