"""Supervisor subgraph for the Deep Research agent.

This module provides the SupervisorAgent class that wraps the research
supervisor workflow (planning, delegation, tool execution) as a standalone
LangGraph subgraph.
"""

import asyncio
from typing import Literal, Optional, Any

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Command

from src.app.agents.open_deep_research.config import (
    MAX_CONCURRENT_RESEARCH_UNITS,
    MAX_RESEARCHER_ITERATIONS,
    RESEARCH_MODEL,
    supervisor_model,
)
from src.app.agents.open_deep_research.researcher_subgraph import ResearcherAgent
from src.app.agents.open_deep_research.state import (
    SupervisorState,
)
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.common.token_limit import is_token_limit_exceeded
from src.app.core.common.utils import (
    get_notes_from_tool_calls,
)


class SupervisorAgent:
    """Lead research supervisor subgraph that plans and delegates research tasks.

    The supervisor analyzes the research brief and decides how to break down
    the research into manageable tasks. It delegates to sub-researchers via
    a compiled researcher subgraph and coordinates the overall research process.

    The supervisor manages its own LLM models and tools internally
    through configurable_model, so the harness tools
    are not used directly by the graph nodes.
    """

    def __init__(self, name: str, tools: list[BaseTool]):
        self.name = name
        self.tools = tools
        self._graph: Optional[CompiledStateGraph] = None
        self._researcher_agent: Optional[ResearcherAgent] = None

    async def compile(self) -> CompiledStateGraph:
        graph_builder = await self._create_graph()
        self._graph = graph_builder.compile(name=self.name)
        logger.info("graph_created", graph_name=self.name, environment=settings.ENVIRONMENT.value)
        return self._graph

    def get_graph(self) -> CompiledStateGraph:
        """Get the compiled supervisor graph instance.

        Returns:
            The compiled supervisor graph.
        """
        if self._graph is None:
            raise ValueError("Supervisor graph has not been compiled yet.")
        return self._graph

    def set_researcher_agent(self, researcher_agent: ResearcherAgent):
        """Set the compiled researcher subgraph used for research delegation.

        Args:
            researcher_agent: The compiled researcher subgraph instance.
        """
        self._researcher_agent = researcher_agent

    async def _supervisor_node(self, state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
        """Lead research supervisor that plans research strategy and delegates to researchers.

        The supervisor analyzes the research brief and decides how to break down the research
        into manageable tasks. It can use think_tool for strategic planning, ConductResearch
        to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.

        Args:
            state: Current supervisor state with messages and research context
            config: Runtime configuration with model settings

        Returns:
            Command to proceed to supervisor_tools for tool execution
        """
        logger.info("node_start", node="_supervisor_node")

        supervisor_messages = state.get("supervisor_messages", [])
        supervisor_model.tools(self.tools)
        response = await supervisor_model.ainvoke(supervisor_messages)

        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )

    async def _supervisor_tools_node(self, state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
        """Execute tools called by the supervisor, including research delegation and strategic thinking.

        This function handles three types of supervisor tool calls:
        1. think_tool - Strategic reflection that continues the conversation
        2. ConductResearch - Delegates research tasks to sub-researchers
        3. ResearchComplete - Signals completion of research phase

        Args:
            state: Current supervisor state with messages and iteration count
            config: Runtime configuration with research limits and model settings

        Returns:
            Command to either continue supervision loop or end research phase
        """
        logger.info("node_start", node="_supervisor_tools_node")
        supervisor_messages = state.get("supervisor_messages", [])
        research_iterations = state.get("research_iterations", 0)
        most_recent_message = supervisor_messages[-1]

        exceeded_allowed_iterations = research_iterations > MAX_RESEARCHER_ITERATIONS
        no_tool_calls = not most_recent_message.tool_calls
        research_complete_tool_call = any(
            tool_call["name"] == "ResearchComplete"
            for tool_call in most_recent_message.tool_calls
        )

        if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )

        all_tool_messages = []
        think_tool_calls = [
            tool_call for tool_call in most_recent_message.tool_calls
            if tool_call["name"] == "think_tool"
        ]

        for tool_call in think_tool_calls:
            reflection_content = tool_call["args"]["reflection"]
            all_tool_messages.append(ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"]
            ))

        conduct_research_calls = [
            tool_call for tool_call in most_recent_message.tool_calls
            if tool_call["name"] == "ConductResearch"
        ]

        update_payload = {"supervisor_messages": []}
        try:
            if conduct_research_calls:
                update_payload = await self._research_tool_call(conduct_research_calls, all_tool_messages)
        except Exception as e:
            if is_token_limit_exceeded(e, RESEARCH_MODEL) or True:
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )

        update_payload["supervisor_messages"] = all_tool_messages
        return Command(
            goto="supervisor",
            update=update_payload
        )

    async def _create_graph(self) -> StateGraph:
        """Build the supervisor subgraph workflow.

        Returns:
            StateGraph: The uncompiled supervisor graph with all nodes and edges.
        """
        try:
            graph_builder = StateGraph(SupervisorState)

            graph_builder.add_node("supervisor", self._supervisor_node)
            graph_builder.add_node("supervisor_tools", self._supervisor_tools_node)

            graph_builder.add_edge(START, "supervisor")

            return graph_builder
        except Exception as e:
            logger.error("supervisor_subgraph_creation_failed", error=str(e))
            raise e

    async def _research_tool_call(self, conduct_research_calls, all_tool_messages) -> dict[str, Any]:
        update_payload = {"supervisor_messages": []}
        allowed_conduct_research_calls = conduct_research_calls[:MAX_CONCURRENT_RESEARCH_UNITS]
        overflow_conduct_research_calls = conduct_research_calls[MAX_CONCURRENT_RESEARCH_UNITS:]

        research_tasks = [
            self._researcher_agent.agent_invoke({
                "researcher_messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, "session_id_placeholder", 1)
            for tool_call in allowed_conduct_research_calls
        ]

        tool_results = await asyncio.gather(*research_tasks)

        for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
            all_tool_messages.append(ToolMessage(
                content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))

        for overflow_call in overflow_conduct_research_calls:
            all_tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {MAX_CONCURRENT_RESEARCH_UNITS} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_call["id"]
            ))

        raw_notes_concat = "\n".join([
            "\n".join(observation.get("raw_notes", []))
            for observation in tool_results
        ])

        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]

        return update_payload
