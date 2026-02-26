"""Deep Research agent module.

This module provides the DeepResearchAgent class that integrates the
multi-subgraph deep research workflow with the project's checkpointing,
Langfuse tracing, and session management infrastructure.
"""

from typing import Any, AsyncGenerator, Optional

from asgiref.sync import sync_to_async
from langchain_core.messages import convert_to_openai_messages
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import StateSnapshot

from src.app.core.guardrails import create_input_guardrail_node, create_output_guardrail_node
from src.app.agents.open_deep_research.deep_researcher import clarify_with_user, write_research_brief, final_report_generation
from src.app.agents.open_deep_research.researcher_subgraph import ResearcherAgent
from src.app.agents.open_deep_research.state import AgentState, AgentInputState, ConductResearch, ResearchComplete
from src.app.agents.open_deep_research.supervisor_subgraph import SupervisorAgent
from src.app.agents.open_deep_research.utils import get_all_tools
from src.app.agents.tools.think_tool import think_tool
from src.app.core.common.config import Environment, settings
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.logging import logger
from src.app.core.common.model.message import Message
from src.app.core.llm.llm_utils import dump_messages, record_llm_error
from src.app.core.memory.memory import bg_update_memory
from src.app.init import langfuse_callback_handler


class DeepResearchAgent:
    """Deep Research agent using supervisor-researcher multi-subgraph architecture.

    This agent conducts multi-step research by:
    1. Clarifying the user's research question
    2. Writing a research brief
    3. Delegating research to parallel sub-researchers via a supervisor
    4. Generating a comprehensive final report

    The deep researcher manages its own LLM models and tools internally
    through hardcoded constants, so the harness tools
    are not used directly by the graph nodes.
    """

    def __init__(self, name: str, checkpointer: AsyncPostgresSaver):
        self.name = name
        self.checkpointer = checkpointer
        self._graph: Optional[CompiledStateGraph] = None
        self._config = {
            "callbacks": [langfuse_callback_handler],
            "metadata": {
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

        lead_researcher_tools = [tool(ConductResearch), tool(ResearchComplete), think_tool]
        self.researcher_subagent = ResearcherAgent("Researcher", get_all_tools())
        self.supervisor_subagent = SupervisorAgent("Supervisor", lead_researcher_tools)

    async def compile(self) -> CompiledStateGraph:
        """Compile all subgraphs and the main deep research graph."""
        try:
            await self.researcher_subagent.compile()
            self.supervisor_subagent.set_researcher_agent(self.researcher_subagent)
            await self.supervisor_subagent.compile()

            graph_builder = self._build_deep_research_graph()
            self._graph = graph_builder.compile(checkpointer=self.checkpointer, name=self.name)

            logger.info(
                "graph_created",
                graph_name=self.name,
                environment=settings.ENVIRONMENT.value,
                has_checkpointer=self.checkpointer is not None,
            )
            return self._graph
        except Exception as e:
            logger.error("deep_research_graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
            raise e

    async def agent_invoke(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        """Invoke the deep research agent with the given messages.

        Args:
            messages: The user messages to process.
            session_id: The session ID for the conversation.
            user_id: The user ID for memory operations.

        Returns:
            list[Message]: The processed response messages.
        """
        config = self._build_invoke_config(session_id, user_id)

        try:
            response = await self._graph.ainvoke(
                input={"messages": dump_messages(messages)},
                config=config,
            )

            if response.get("messages"):
                bg_update_memory(user_id, convert_to_openai_messages(response["messages"]), config["metadata"])

            return process_messages(response["messages"])

        except Exception as e:
            record_llm_error("deep_research", self.name)
            if settings.ENVIRONMENT == Environment.DEVELOPMENT:
                raise e
            logger.exception("deep_research_invoke_failed", session_id=session_id, error=str(e))
            return []

    async def agent_invoke_stream(
        self, messages: list[Message], session_id: str, user_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the deep research agent response token by token.

        Args:
            messages: The messages to send to the agent.
            session_id: The session ID for the conversation.
            user_id: The user ID for the conversation.

        Yields:
            str: Tokens of the agent response.
        """
        config = self._build_invoke_config(session_id, user_id)

        try:
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages)},
                config,
                stream_mode="messages",
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    logger.error("error_processing_token", error=str(token_error), session_id=session_id)
                    continue

            state: StateSnapshot = await sync_to_async(self._graph.get_state)(config=config)
            if state.values and "messages" in state.values:
                bg_update_memory(user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"])

        except Exception as stream_error:
            record_llm_error("deep_research", self.name)
            logger.error("deep_research_stream_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    def _build_invoke_config(self, session_id: str, user_id: Optional[int] = None) -> dict:
        config = self._config.copy()
        config["configurable"] = {"thread_id": session_id}
        config["metadata"]["user_id"] = user_id
        config["metadata"]["session_id"] = session_id
        return config

    def _build_deep_research_graph(self) -> StateGraph:
        """Build the complete deep research workflow graph (uncompiled).

        Returns:
            StateGraph: The uncompiled deep research graph builder.
        """
        input_guardrail = create_input_guardrail_node(next_node="clarify_with_user")
        output_guardrail = create_output_guardrail_node()

        deep_researcher_builder = StateGraph(AgentState, input=AgentInputState)

        # Guardrail nodes wrap the entire research workflow
        deep_researcher_builder.add_node("input_guardrail", input_guardrail, ends=["clarify_with_user", END])
        deep_researcher_builder.add_node("output_guardrail", output_guardrail)

        # Main workflow nodes for the complete research process
        deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
        deep_researcher_builder.add_node("write_research_brief", write_research_brief)
        deep_researcher_builder.add_node("research_supervisor", self.supervisor_subagent.get_graph())
        deep_researcher_builder.add_node("final_report_generation", final_report_generation)

        # Workflow edges: input_guardrail → research pipeline → output_guardrail
        deep_researcher_builder.add_edge(START, "input_guardrail")
        deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
        deep_researcher_builder.add_edge("final_report_generation", "output_guardrail")
        deep_researcher_builder.add_edge("output_guardrail", END)

        return deep_researcher_builder
