from abc import abstractmethod
from typing import Optional, Any, AsyncGenerator

from asgiref.sync import sync_to_async
from langchain_core.messages import ToolMessage
from langchain_core.messages import convert_to_openai_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import StateSnapshot, Command

from src.app.core.common.config import settings, Environment
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.logging import logger
from src.app.core.common.model.graph import GraphState
from src.app.core.common.model.message import Message
from src.app.core.llm.llm_utils import dump_messages
from src.app.core.mcp.session_manager import get_mcp_session_manager
from src.app.core.mcp.mcp_utils import handle_mcp_tool_call
from src.app.core.memory.memory import get_relevant_memory, bg_update_memory
from src.app.init import langfuse_callback_handler


class AgentAbstract:
    """Abstract base class for all agents in the framework."""
    _graph: Optional[CompiledStateGraph] = None

    def __init__(self, name: str, agent_name: str, tools: list[BaseTool], checkpointer: AsyncPostgresSaver = None):
        self.checkpointer = checkpointer
        self.name = name
        self.agent_name = agent_name
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.mcp_tools_by_name = {}
        self.config = {
            "callbacks": [langfuse_callback_handler],
            "metadata": {
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

    async def compile(self) -> Optional[CompiledStateGraph]:
        """Compile the graph and prepare for execution."""
        graph_ = await self._create_graph()
        self._graph = await self._create_compiled_graph(graph_)
        return self._graph

    async def agent_invoke(
        self,
        agent_input: dict[str, Any],
        session_id: str,
        user_id: Optional[int] = None,
    ) -> list[Message] | list[Any]:
        config = await self._create_config(session_id, user_id)
        try:
            response = await self._graph.ainvoke(input=agent_input, config=config)
            openai_style_messages = convert_to_openai_messages(response["messages"])
            return [
                # keep just assistant and user messages
                Message(role=message["role"], content=str(message["content"]))
                for message in openai_style_messages
                if message["role"] in ["assistant", "user"] and message["content"]
            ]
        except Exception as e:
            if settings.ENVIRONMENT == Environment.DEVELOPMENT:
                raise e
            logger.error(f"Error getting response: {str(e)}")
            return []

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

    async def _create_config(self, session_id, user_id):
        config = self.config.copy()
        config["configurable"] = {"thread_id": session_id}
        config["metadata"]["user_id"] = user_id
        config["metadata"]["session_id"] = session_id
        return config

    async def agent_invoke_stream(
        self, messages: list[Message], session_id: str, user_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """

        config = await self._create_config(session_id, user_id)
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
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue

            # After streaming completes, get final state and update memory in background
            state: StateSnapshot = await sync_to_async(self._graph.get_state)(config=config)
            if state.values and "messages" in state.values:
                bg_update_memory(user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"])

        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

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
            # Existing outer exception handling if needed

        return Command(update={"messages": outputs}, goto="chat")

    async def _create_compiled_graph(self, graph_builder: StateGraph) -> Optional[CompiledStateGraph]:
        """Create and compile the LangGraph workflow with nodes and checkpointer.

        Args:
            chat_node_fn: The chat node function.
            tool_call_node_fn: The tool call node function.

        Returns:
            Optional[CompiledStateGraph]: The compiled graph instance, or None if creation fails in production.
        """
        try:
            await self._load_mcp_tools()

            compiled_graph = graph_builder.compile(
                checkpointer=self.checkpointer, name=f"{self.name}"
            )

            logger.info(
                "graph_created",
                graph_name=f"{self.name}",
                environment=settings.ENVIRONMENT.value,
                has_checkpointer=self.checkpointer is not None,
            )
            return compiled_graph
        except Exception as e:
            logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
            raise e

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return process_messages(state.values["messages"]) if state.values else []

    @abstractmethod
    def _create_graph(self):
        raise NotImplementedError("create_graph not implemented")
