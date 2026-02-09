from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import RunnableConfig, Command
from src.app.agents.example1 import load_system_prompt

from src.app.core.agentic.agent_base import AgentAbstract
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.common.metrics import llm_inference_duration_seconds
from src.app.core.common.model.graph import GraphState
from src.app.core.llm.llm import LLMService
from src.app.core.llm.llm_utils import dump_messages, prepare_messages, process_llm_response


class AgentExample1 (AgentAbstract):
    """Example agent to demonstrate the agentic framework."""
    _graph: Optional[CompiledStateGraph] = None

    def __init__(self, name, llm_service: LLMService, tools: list, checkpointer: AsyncPostgresSaver):
        super().__init__(name, llm_service, tools, checkpointer)

    async def _chat_node(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state: The current state of the conversation.
            config: The configuration for the node execution.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        # Get the current LLM instance for metrics
        current_llm = self.llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.DEFAULT_LLM_MODEL
        )

        system_prompt = load_system_prompt(long_term_memory=state.long_term_memory)
        messages = prepare_messages(state.messages, current_llm, system_prompt)

        try:
            # Use LLM service with automatic retries and circular fallback
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message = await self.llm_service.call(dump_messages(messages))
            response_message = process_llm_response(response_message)
            logger.info(
                "llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
                environment=settings.ENVIRONMENT.value,
            )

            # Determine next node based on whether there are tool calls
            if response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.error(
                "llm_call_failed_all_models",
                session_id=config["configurable"]["thread_id"],
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"failed to get llm response after trying all models: {str(e)}")


    async def _create_graph(self) -> StateGraph:
        try:
            graph_builder = StateGraph(GraphState)
            graph_builder.add_node("chat", self._chat_node, ends=["tool_call", END])
            graph_builder.add_node("tool_call", self._tool_call_node, ends=["chat"])
            graph_builder.set_entry_point("chat")
            graph_builder.set_finish_point("chat")
            return graph_builder
        except Exception as e:
            logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
            raise e


