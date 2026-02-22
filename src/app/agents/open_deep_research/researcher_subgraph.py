"""Researcher subgraph extending AgentAbstract.

This module provides the ResearcherSubgraph class that wraps the individual
researcher workflow (research, tool execution, compression) in the project's
AgentAbstract class structure.
"""

import asyncio
from typing import Literal, Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Command

from src.app.agents.open_deep_research.config import (
    COMPRESSION_MODEL,
    COMPRESSION_MODEL_MAX_TOKENS,
    MAX_REACT_TOOL_CALLS,
    MAX_STRUCTURED_OUTPUT_RETRIES,
    RESEARCH_MODEL,
    RESEARCH_MODEL_MAX_TOKENS,
    configurable_model,
)
from src.app.agents.open_deep_research.prompts import (
    compress_research_simple_human_message,
    compress_research_system_prompt,
    research_system_prompt,
)
from src.app.agents.open_deep_research.state import (
    ResearcherOutputState,
    ResearcherState,
)
from src.app.agents.open_deep_research.utils import (
    anthropic_websearch_called,
    openai_websearch_called,
    remove_up_to_last_ai_message,
)
from src.app.core.agentic.agent_base import AgentAbstract
from src.app.core.common.logging import logger
from src.app.core.common.token_limit import is_token_limit_exceeded
from src.app.core.common.utils import get_api_key_for_model, get_today_str, execute_tool_safely, execute_tools


class ResearcherAgent(AgentAbstract):
    """Individual researcher subgraph that conducts focused research on specific topics.

    This subgraph is given a specific research topic by the supervisor and uses
    available tools (search, think_tool) to gather comprehensive information,
    then compresses findings into a concise summary.

    The researcher manages its own LLM models and tools internally
    through configurable_model, so the harness tools
    are not used directly by the graph nodes.
    """

    _graph: Optional[CompiledStateGraph] = None


    async def _researcher_node(self, state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
        """Individual researcher that conducts focused research on specific topics.

        This researcher is given a specific research topic by the supervisor and uses
        available tools (search, think_tool) to gather comprehensive information.
        It can use think_tool for strategic planning between searches.

        Args:
            state: Current researcher state with messages and topic context
            config: Runtime configuration with model settings and tool availability

        Returns:
            Command to proceed to researcher_tools for tool execution
        """
        logger.info("node_start", node="researcher_node", tool_call_iterations=state.get("tool_call_iterations", 0))
        researcher_messages = state.get("researcher_messages", [])

        tools = self.tools
        if len(tools) == 0:
            raise ValueError(
                "No tools found to conduct research: Please configure your search API."
            )

        research_model_config = {
            "model": RESEARCH_MODEL,
            "max_tokens": RESEARCH_MODEL_MAX_TOKENS,
            "api_key": get_api_key_for_model(RESEARCH_MODEL, config),
            "tags": ["langsmith:nostream"]
        }

        researcher_prompt = research_system_prompt.format(date=get_today_str())

        research_model = (
            configurable_model
            .bind_tools(tools)
            .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
            .with_config(research_model_config)
        )

        messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
        response = await research_model.ainvoke(messages)

        return Command(
            goto="researcher_tools",
            update={
                "researcher_messages": [response],
                "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
            }
        )

    async def _researcher_tools_node(self, state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
        """Execute tools called by the researcher, including search tools and strategic thinking.

        Args:
            state: Current researcher state with messages and iteration count
            config: Runtime configuration with research limits and tool settings

        Returns:
            Command to either continue research loop or proceed to compression
        """
        logger.info("node_start", node="researcher_tools", tool_call_iterations=state.get("tool_call_iterations", 0))
        researcher_messages = state.get("researcher_messages", [])
        most_recent_message = researcher_messages[-1]

        has_tool_calls = bool(most_recent_message.tool_calls)
        has_native_search = (
            openai_websearch_called(most_recent_message) or
            anthropic_websearch_called(most_recent_message)
        )

        if not has_tool_calls and not has_native_search:
            logger.info("has_native_search", tool_count=len(most_recent_message.tool_calls))
            return Command(goto="compress_research")

        tools = self.tools
        tools_by_name = {
            tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
            for tool in tools
        }

        tool_outputs = await execute_tools(config, most_recent_message, tools_by_name)

        exceeded_iterations = state.get("tool_call_iterations", 0) >= MAX_REACT_TOOL_CALLS
        research_complete_called = any(
            tool_call["name"] == "ResearchComplete"
            for tool_call in most_recent_message.tool_calls
        )

        if exceeded_iterations or research_complete_called:
            return Command(
                goto="compress_research",
                update={"researcher_messages": tool_outputs}
            )

        return Command(
            goto="researcher",
            update={"researcher_messages": tool_outputs}
        )


    async def _compress_research_node(self, state: ResearcherState, config: RunnableConfig):
        """Compress and synthesize research findings into a concise, structured summary.

        Args:
            state: Current researcher state with accumulated research messages
            config: Runtime configuration with compression model settings

        Returns:
            Dictionary containing compressed research summary and raw notes
        """
        logger.info("node_start", node="_compress_research_node", tool_call_iterations=state.get("tool_call_iterations", 0))
        model_config = {
            "model": COMPRESSION_MODEL,
            "max_tokens": COMPRESSION_MODEL_MAX_TOKENS,
            "api_key": get_api_key_for_model(COMPRESSION_MODEL, config),
            "tags": ["langsmith:nostream"]
        }
        synthesizer_model = configurable_model.with_config(model_config)

        researcher_messages = state.get("researcher_messages", [])
        researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

        synthesis_attempts = 0
        max_attempts = 3

        while synthesis_attempts < max_attempts:
            try:
                compression_prompt = compress_research_system_prompt.format(date=get_today_str())
                messages = [SystemMessage(content=compression_prompt)] + researcher_messages

                response = await synthesizer_model.ainvoke(messages)

                raw_notes_content = "\n".join([
                    str(message.content)
                    for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
                ])

                return {
                    "compressed_research": str(response.content),
                    "raw_notes": [raw_notes_content]
                }

            except Exception as e:
                synthesis_attempts += 1

                if is_token_limit_exceeded(e, RESEARCH_MODEL):
                    researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                    continue

                continue

        raw_notes_content = "\n".join([
            str(message.content)
            for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
        ])

        return {
            "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
            "raw_notes": [raw_notes_content]
        }

    async def _create_graph(self) -> StateGraph:
        """Build the researcher subgraph workflow.

        Returns:
            StateGraph: The uncompiled researcher graph with all nodes and edges.
        """
        try:
            graph_builder = StateGraph(
                ResearcherState,
                output=ResearcherOutputState,
            )

            graph_builder.add_node("researcher", self._researcher_node)
            graph_builder.add_node("researcher_tools", self._researcher_tools_node)
            graph_builder.add_node("compress_research", self._compress_research_node)

            graph_builder.add_edge(START, "researcher")
            graph_builder.add_edge("compress_research", END)

            return graph_builder
        except Exception as e:
            logger.error("researcher_subgraph_creation_failed", error=str(e))
            raise e
