import os
import time
from datetime import datetime
from collections.abc import Sequence
from typing import Optional, Any, AsyncGenerator

from asgiref.sync import sync_to_async
from langchain_core.messages import SystemMessage, ToolMessage, convert_to_openai_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import RunnableConfig, Command, StateSnapshot

from src.app.core.graph import END, GraphBuilder, StateGraphCompiled
from src.app.core.middleware import (
    AgentContext,
    AgentMiddleware,
    AgentPipeline,
    build_invoke_config,
)
from src.app.core.common.config import settings
from src.app.core.common.graph_utils import process_messages
from src.app.core.common.logging import logger
from src.app.core.metrics.metrics import tool_executions_total
from src.app.core.common.model.graph import GraphState
from src.app.core.common.model.message import Message
from src.app.core.llm.llm_utils import dump_messages, process_llm_response, record_llm_error
from src.app.core.fault_tolerance import (
    create_chat_node_error_handler,
    create_tool_node_error_handler,
    get_llm_retry_policy,
    get_llm_timeout_policy,
    get_tool_retry_policy,
    get_tool_timeout_policy,
)
from src.app.core.context import truncate_tool_call_if_too_long
from src.app.core.mcp.mcp_utils import handle_mcp_tool_call
from src.app.core.mcp.session_manager import get_mcp_session_manager
from src.app.core.memory import memory_service

from src.app.core.llm.factory import make_chat_model


chatbot_model = make_chat_model(
    model=f"openai:{settings.DEFAULT_LLM_MODEL}",
    api_key=settings.OPENAI_API_KEY,
    max_tokens=settings.MAX_TOKENS,
)


def build_chatbot_trace_metadata(ctx: AgentContext) -> dict[str, Any]:
    return dict(ctx.metadata.get("trace_metadata", {}))


def build_chatbot_trace_output(ctx: AgentContext, result: list[Message]) -> dict[str, Any]:
    execution_time = time.time() - ctx.metadata.get("_trace_start_time", time.time())
    graph_result = ctx.metadata.get("graph_result")
    answer = ""
    if graph_result and graph_result.get("messages"):
        openai_messages = convert_to_openai_messages(graph_result["messages"])
        assistant_messages = [m for m in openai_messages if m.get("role") == "assistant" and m.get("content")]
        if assistant_messages:
            answer = str(assistant_messages[-1]["content"])
    elif result:
        answer = result[-1].content
    return {
        "answer": answer,
        "message_count": len(result),
        "execution_time": execution_time,
    }


class AgentChatbot:
    """Example agent to demonstrate the agentic framework."""

    def __init__(
        self,
        name: str,
        tools: list[BaseTool],
        checkpointer: AsyncPostgresSaver,
        middlewares: Sequence[AgentMiddleware],
    ):
        self.name = name
        self.checkpointer = checkpointer
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.mcp_tools_by_name: dict[str, BaseTool] = {}
        self._graph: Optional[StateGraphCompiled] = None
        self._last_trace_id: Optional[str] = None
        self._pipeline = AgentPipeline(
            middlewares=middlewares,
            invoke_fn=self._core_invoke,
        )

    @property
    def last_trace_id(self) -> Optional[str]:
        return self._last_trace_id

    async def compile(self) -> StateGraphCompiled:
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
        """Invoke the agent through the middleware pipeline."""
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
                "model_name": settings.DEFAULT_LLM_MODEL,
                "trace_metadata": {
                    "service": "chatbot",
                    "model": settings.DEFAULT_LLM_MODEL,
                },
            },
        )
        result = await self._pipeline.run(ctx)
        self._last_trace_id = ctx.metadata.get("trace_id")
        return result

    async def _core_invoke(self, ctx: AgentContext) -> list[Message]:
        """Core graph invocation without cross-cutting concerns."""
        long_term_memory = ctx.metadata.get("long_term_memory", "")
        agent_input = {"messages": dump_messages(ctx.messages), "long_term_memory": long_term_memory}

        response = await self._graph.ainvoke(
            agent_input,
            ctx.config,
        )
        ctx.metadata["graph_result"] = response
        openai_style_messages = convert_to_openai_messages(response["messages"])
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

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
        config = build_invoke_config(session_id, user_id, self.name)
        relevant_memory = (
            await memory_service.search(user_id, messages[-1].content)
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
                memory_service.schedule_add(
                    user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"]
                )

        except Exception as stream_error:
            record_llm_error(settings.DEFAULT_LLM_MODEL, self.name)
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
        """Process tool calls from the last message."""
        outputs = []
        manager = self._pipeline.manager
        ctx = manager.active_ctx

        try:
            for tool_call in state.messages[-1].tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if ctx:
                    tool_args = await manager.run_before_tool_call(
                        ctx, tool_name=tool_name, tool_args=tool_args,
                    )

                if tool_name in self.tools_by_name:
                    try:
                        tool_result = await self.tools_by_name[tool_name].ainvoke(tool_args)
                        tool_executions_total.labels(tool_name=tool_name, status="success").inc()
                    except Exception as tool_error:
                        tool_executions_total.labels(tool_name=tool_name, status="error").inc()
                        raise tool_error

                    if ctx:
                        tool_result = await manager.run_after_tool_call(
                            ctx, tool_name=tool_name, tool_result=tool_result,
                        )

                    outputs.append(truncate_tool_call_if_too_long(
                        ToolMessage(
                            content=tool_result,
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    ))
                elif tool_name in self.mcp_tools_by_name:
                    tool_fn = self.mcp_tools_by_name[tool_name]
                    tool_message = await handle_mcp_tool_call(
                        tool_fn=tool_fn,
                        tool_call=tool_call,
                        tool_name=tool_name,
                        max_retries=1,
                        on_reconnect=self._load_mcp_tools,
                    )
                    outputs.append(truncate_tool_call_if_too_long(tool_message))

        except Exception as e:
            logger.error("tool_call_processing_failed", error=str(e))
            raise

        return Command(update={"messages": outputs}, goto="chat")

    async def _chat_node(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response."""
        manager = self._pipeline.manager
        ctx = manager.active_ctx

        messages = state.messages
        if ctx:
            messages = await manager.run_before_model_call(
                ctx, messages=messages, model_name=settings.DEFAULT_LLM_MODEL,
            )

        system_prompt = load_system_prompt(long_term_memory=state.long_term_memory)
        prepared = [SystemMessage(content=system_prompt)] + list(messages)

        model = chatbot_model.bind_tools(self._get_all_tools())

        response_message = await model.ainvoke(prepared, config)

        if ctx:
            response_message = await manager.run_after_model_call(
                ctx, response=response_message, model_name=settings.DEFAULT_LLM_MODEL,
            )

        response_message = process_llm_response(response_message)
        logger.info(
            "llm_response_generated",
            session_id=config["configurable"]["thread_id"],
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

        goto = "tool_call" if response_message.tool_calls else END

        return Command(update={"messages": [response_message]}, goto=goto)

    async def _create_graph(self) -> GraphBuilder:
        try:
            graph_builder = GraphBuilder(GraphState).set_middleware_manager(self._pipeline.manager)
            graph_builder.add_node(
                "chat",
                self._chat_node,
                ends=["tool_call", END],
                retry_policy=get_llm_retry_policy(),
                timeout=get_llm_timeout_policy(),
                error_handler=create_chat_node_error_handler(
                    agent_name=self.name,
                    model_name=settings.DEFAULT_LLM_MODEL,
                    fallback_goto=END,
                ),
            )
            graph_builder.add_node(
                "tool_call",
                self._tool_call_node,
                ends=["chat"],
                retry_policy=get_tool_retry_policy(),
                timeout=get_tool_timeout_policy(),
                error_handler=create_tool_node_error_handler(
                    agent_name=self.name,
                    fallback_goto="chat",
                ),
            )
            graph_builder.set_entry_point("chat")
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
