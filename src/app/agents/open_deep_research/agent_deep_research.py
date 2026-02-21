"""Deep Research agent adapted to the project's agent harness.

This module provides the DeepResearchAgent class that extends AgentAbstract
to integrate the multi-subgraph deep research workflow with the project's
checkpointing, Langfuse tracing, and session management infrastructure.
"""

from typing import Any, AsyncGenerator, Optional

from asgiref.sync import sync_to_async
from langchain_core.messages import convert_to_openai_messages
from langchain_core.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import StateSnapshot

from src.app.agents.open_deep_research.deep_researcher import clarify_with_user, write_research_brief, final_report_generation
from src.app.agents.open_deep_research.researcher_subgraph import ResearcherAgent
from src.app.agents.open_deep_research.supervisor_subgraph import SupervisorAgent
from src.app.agents.open_deep_research.state import AgentState, AgentInputState, ConductResearch, ResearchComplete
from src.app.agents.open_deep_research.utils import get_all_tools
from src.app.agents.tools.think_tool import think_tool
from src.app.core.agentic.agent_base import AgentAbstract
from src.app.core.common.config import Environment, settings
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.logging import logger
from src.app.core.common.model.message import Message
from src.app.core.llm.llm import LLMService
from src.app.core.llm.llm_utils import dump_messages
from src.app.core.memory.memory import bg_update_memory, get_relevant_memory


class DeepResearchAgent(AgentAbstract):
    """Deep Research agent using supervisor-researcher multi-subgraph architecture.

    This agent conducts multi-step research by:
    1. Clarifying the user's research question
    2. Writing a research brief
    3. Delegating research to parallel sub-researchers via a supervisor
    4. Generating a comprehensive final report

    The deep researcher manages its own LLM models and tools internally
    through hardcoded constants, so the harness LLMService and tools
    are not used directly by the graph nodes.
    """

    _graph: Optional[CompiledStateGraph] = None

    def __init__(self, name: str, llm_service: LLMService, checkpointer: AsyncPostgresSaver):
        super().__init__(name, llm_service, [], checkpointer)
        lead_researcher_tools = [tool(ConductResearch), tool(ResearchComplete), think_tool]
        self.researcher_subagent = ResearcherAgent("Researcher", llm_service, get_all_tools(), None)
        self.supervisor_subagent = SupervisorAgent("Supervisor", llm_service, lead_researcher_tools, None)

    async def _create_graph(self) -> StateGraph:
        """Build the deep research multi-subgraph workflow.

        Compiles the researcher and supervisor subgraphs first, then builds
        the main deep research graph using their compiled graphs as nodes.

        Returns:
            StateGraph: The uncompiled deep research graph with all nodes and edges.
        """
        try:
            await self.researcher_subagent.compile()
            self.supervisor_subagent.set_researcher_agent(self.researcher_subagent)
            await self.supervisor_subagent.compile()

            return build_deep_research_graph(self.supervisor_subagent._graph)
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

        Adapts the harness invoke pattern to the deep researcher's AgentInputState
        which only accepts messages (no long_term_memory field).

        Args:
            messages: The user messages to process.
            session_id: The session ID for the conversation.
            user_id: The user ID for memory operations.

        Returns:
            list[Message]: The processed response messages.
        """
        config = await self._create_config(session_id, user_id)

        try:
            response = await self._graph.ainvoke(
                input={"messages": dump_messages(messages)},
                config=config,
            )

            # Update long-term memory in background
            if response.get("messages"):
                bg_update_memory(user_id, convert_to_openai_messages(response["messages"]), config["metadata"])

            return process_messages(response["messages"])

        except Exception as e:
            if settings.ENVIRONMENT == Environment.DEVELOPMENT:
                raise e
            logger.exception("deep_research_invoke_failed", session_id=session_id, error=str(e))
            return []

    async def agent_invoke_stream(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the deep research agent response.

        Streams tokens from the deep research graph. Because the deep researcher
        runs a multi-step workflow, intermediate tokens from all phases are streamed.

        Args:
            messages: The user messages to process.
            session_id: The session ID for the conversation.
            user_id: The user ID for memory operations.

        Yields:
            str: Tokens of the agent response.
        """
        config = await self._create_config(session_id, user_id)

        try:
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages)},
                config,
                stream_mode="messages",
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    logger.error(
                        "deep_research_stream_token_error",
                        error=str(token_error),
                        session_id=session_id,
                    )
                    continue

            # After streaming completes, update memory in background
            state: StateSnapshot = await sync_to_async(self._graph.get_state)(config=config)
            if state.values and "messages" in state.values:
                bg_update_memory(user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"])

        except Exception as stream_error:
            logger.exception(
                "deep_research_stream_failed",
                error=str(stream_error),
                session_id=session_id,
            )
            raise stream_error

def build_deep_research_graph(compiled_supervisor_subgraph: CompiledStateGraph) -> StateGraph:
    """Build the complete deep research workflow graph (uncompiled).

    Creates the main deep research StateGraph with all nodes and edges.
    The subgraphs (supervisor, researcher) are compiled by the DeepResearchAgent
    and the supervisor's compiled graph is passed in as a parameter.

    Args:
        compiled_supervisor_subgraph: The compiled supervisor subgraph to embed as a node.

    Returns:
        StateGraph: The uncompiled deep research graph builder.
    """
    deep_researcher_builder = StateGraph(
        AgentState,
        input=AgentInputState,
    )

    # Add main workflow nodes for the complete research process
    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)
    deep_researcher_builder.add_node("research_supervisor", compiled_supervisor_subgraph)
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)

    # Define main workflow edges for sequential execution
    deep_researcher_builder.add_edge(START, "clarify_with_user")
    deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
    deep_researcher_builder.add_edge("final_report_generation", END)

    return deep_researcher_builder
