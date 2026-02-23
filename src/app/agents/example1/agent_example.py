import os
from datetime import datetime
from typing import Optional, Any, AsyncGenerator

from asgiref.sync import sync_to_async
from langchain_core.messages import ToolMessage, convert_to_openai_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import RunnableConfig, Command, StateSnapshot

from src.app.core.common.config import settings, Environment
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.logging import logger
from src.app.core.common.metrics import llm_inference_duration_seconds
from src.app.core.common.model.graph import GraphState
from src.app.core.common.model.message import Message
from src.app.core.llm.llm import LLMRegistry
from src.app.core.llm.llm_utils import dump_messages, prepare_messages, process_llm_response
from src.app.core.mcp.mcp_utils import handle_mcp_tool_call
from src.app.core.mcp.session_manager import get_mcp_session_manager
from src.app.core.memory.memory import get_relevant_memory, bg_update_memory
from src.app.init import langfuse_callback_handler


class AgentExample1:
    """Example agent to demonstrate the agentic framework."""

    def __init__(self, name: str, agent_name: str, tools: list[BaseTool], checkpointer: AsyncPostgresSaver):
        self.name = name
        self.agent_name = agent_name
        self.checkpointer = checkpointer
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.mcp_tools_by_name: dict[str, BaseTool] = {}
        self._graph: Optional[CompiledStateGraph] = None
        self._config = {
            "callbacks": [langfuse_callback_handler],
            "metadata": {
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

    async def compile(self) -> CompiledStateGraph:
        """Compile the graph and prepare for execution."""
        await self._load_mcp_tools()
        graph_builder = await self._create_graph()
        self._graph = graph_builder.compile(checkpointer=self.checkpointer, name=self.name)
        logger.info(
            "graph_created",
            graph_name=self.name,
            environment=settings.ENVIRONMENT.value,
            has_checkpointer=self.checkpointer is not None,
        )
        return self._graph

    async def agent_invoke(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        relevant_memory = (await get_relevant_memory(user_id, messages[-1].content)) or "No relevant memory found."
        agent_input = {"messages": dump_messages(messages), "long_term_memory": relevant_memory}
        config = self._build_invoke_config(session_id, user_id)

        try:
            response = await self._graph.ainvoke(input=agent_input, config=config)
            openai_style_messages = convert_to_openai_messages(response["messages"])
            result = [
                Message(role=message["role"], content=str(message["content"]))
                for message in openai_style_messages
                if message["role"] in ["assistant", "user"] and message["content"]
            ]
        except Exception as e:
            if settings.ENVIRONMENT == Environment.DEVELOPMENT:
                raise e
            logger.exception("agent_invoke_failed", session_id=session_id, error=str(e))
            return []

        messages_dic = [dict(role=m.role, content=str(m.content)) for m in result]
        bg_update_memory(user_id, messages_dic, {"session_id": session_id, "agent_name": self.name, "user_id": user_id})
        return result

    async def agent_invoke_stream(
        self, messages: list[Message], session_id: str, user_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the agent response token by token.

        Args:
            messages: The messages to send to the LLM.
            session_id: The session ID for the conversation.
            user_id: The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = self._build_invoke_config(session_id, user_id)
        relevant_memory = (
            await get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."

        try:
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages), "long_term_memory": relevant_memory},
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
            logger.error("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given session.

        Args:
            session_id: The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return process_messages(state.values["messages"]) if state.values else []

    def _build_invoke_config(self, session_id: str, user_id: Optional[int] = None) -> dict:
        config = self._config.copy()
        config["configurable"] = {"thread_id": session_id}
        config["metadata"]["user_id"] = user_id
        config["metadata"]["session_id"] = session_id
        return config

    def _get_all_tools(self) -> list[BaseTool]:
        """Get all available tools including MCP tools."""
        return self.tools + list(self.mcp_tools_by_name.values())

    async def _load_mcp_tools(self):
        """Load tools from persistent MCP sessions."""
        mcp_tools = []

        if settings.MCP_ENABLED:
            try:
                mcp_manager = get_mcp_session_manager()
                resource = mcp_manager.get_resource()
                mcp_tools = resource.tools
                logger.info("mcp_tools_loaded", tool_count=len(mcp_tools))
            except RuntimeError as e:
                logger.warning("mcp_not_initialized", error=str(e))
            except Exception as e:
                logger.error("mcp_tools_load_failed", error=str(e))

        self.mcp_tools_by_name = {tool.name: tool for tool in mcp_tools}
        all_tools_count = len(mcp_tools) + len(self.tools)
        logger.info("tools_loaded", total_count=all_tools_count, mcp_count=len(mcp_tools), builtin_count=len(self.tools))

    async def _tool_call_node(self, state: GraphState) -> Command:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        outputs = []
        try:
            for tool_call in state.messages[-1].tool_calls:
                tool_name = tool_call["name"]

                if tool_name in self.tools_by_name:
                    tool_result = await self.tools_by_name[tool_name].ainvoke(tool_call["args"])
                    outputs.append(
                        ToolMessage(
                            content=tool_result,
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                elif tool_name in self.mcp_tools_by_name:
                    tool_fn = self.mcp_tools_by_name[tool_name]
                    tool_message = await handle_mcp_tool_call(
                        tool_fn=tool_fn,
                        tool_call=tool_call,
                        tool_name=tool_name,
                        max_retries=1,
                        on_reconnect=self._load_mcp_tools,
                    )
                    outputs.append(tool_message)

        except Exception as e:
            logger.error("tool_call_processing_failed", error=str(e))

        return Command(update={"messages": outputs}, goto="chat")

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
