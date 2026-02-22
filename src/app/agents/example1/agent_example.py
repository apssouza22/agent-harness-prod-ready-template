import os
from datetime import datetime
from typing import Optional, Any

from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import RunnableConfig, Command

from src.app.core.agentic.agent_base import AgentAbstract
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.common.metrics import llm_inference_duration_seconds
from src.app.core.common.model.graph import GraphState
from src.app.core.common.model.message import Message
from src.app.core.llm.llm import LLMRegistry
from src.app.core.llm.llm_utils import dump_messages, prepare_messages, process_llm_response
from src.app.core.memory.memory import get_relevant_memory, bg_update_memory


class AgentExample1(AgentAbstract):
    """Example agent to demonstrate the agentic framework."""
    _graph: Optional[CompiledStateGraph] = None

    def __init__(self, name: str, agent_name: str, tools: list[BaseTool], checkpointer: AsyncPostgresSaver):
        super().__init__(name, agent_name, tools, checkpointer)

    async def agent_invoke(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        relevant_memory = (await get_relevant_memory(user_id, messages[-1].content)) or "No relevant memory found."
        cmd = {"messages": dump_messages(messages), "long_term_memory": relevant_memory}
        messages = await super().agent_invoke(cmd, session_id, user_id)
        messages_dic = [dict(role=message.role, content=str(message.content)) for message in messages]
        bg_update_memory(user_id, messages_dic, {"session_id": session_id, "agent_name": self.name, "user_id": user_id})
        return messages

    async def _chat_node(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state: The current state of the conversation.
            config: The configuration for the node execution.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        model_name = settings.DEFAULT_LLM_MODEL
        llm = LLMRegistry.get(model_name, self.agent_name)

        system_prompt = load_system_prompt(long_term_memory=state.long_term_memory)
        messages = prepare_messages(state.messages, llm, system_prompt)

        model = (
            llm
            .bind_tools(self._get_all_tools())
            .with_retry(stop_after_attempt=3)
        )

        try:
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message = await model.ainvoke(dump_messages(messages), config)

            response_message = process_llm_response(response_message)
            logger.info(
                "llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
                environment=settings.ENVIRONMENT.value,
            )

            goto = "tool_call" if response_message.tool_calls else END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.error(
                "llm_call_failed",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise

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


def load_system_prompt(**kwargs):
    """Load the system prompt from the file."""
    with open(os.path.join(os.path.dirname(__file__), "system.md"), "r") as f:
        return f.read().format(
            agent_name=settings.PROJECT_NAME + " Agent",
            current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs,
        )
