import uuid
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

from src.app.core.common.config import settings
from src.app.core.common.logging import logger


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracking MCP operations.

    Returns:
        str: A unique correlation ID in UUID format.
    """
    return str(uuid.uuid4())


class Resource(BaseModel):
    tools: list[StructuredTool]
    sessions: list[ClientSession]

    class Config:
        arbitrary_types_allowed = True


@asynccontextmanager
async def mcp_sse_client(
    mcp_host: str = "http://localhost:7001",
    timeout: int = 2,
    correlation_id: Optional[str] = None,
) -> AsyncGenerator[ClientSession]:
    """
    Creates and initializes an MCP client session over SSE.

    Establishes an SSE connection to the MCP server and yields an initialized
    `ClientSession` for communication.

    Args:
        mcp_host: The MCP server host URL.
        timeout: Connection timeout in seconds.
        correlation_id: Optional correlation ID for tracking this operation.

    Yields:
        ClientSession: An initialized MCP client session.
    """
    try:
        logger.info("mcp_connection_initiated", correlation_id=correlation_id, host=mcp_host)
        async with sse_client(
            f"{mcp_host}/sse",
            timeout=timeout  # in seconds
        ) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("mcp_session_initialized", correlation_id=correlation_id, host=mcp_host)
                yield session
    finally:
        logger.info("mcp_session_closed", correlation_id=correlation_id, host=mcp_host)


class MCPSessionManager:
    """Manages MCP client sessions for the application lifetime."""

    def __init__(self):
        self._exit_stack: Optional[AsyncExitStack] = None
        self._resource: Optional[Resource] = None
        self._initialized: bool = False

    async def initialize(self) -> Resource:
        """Initialize MCP sessions and load tools."""
        if self._initialized:
            return self._resource

        init_correlation_id = generate_correlation_id()
        logger.info("mcp_initialization_started", correlation_id=init_correlation_id)

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        tools = []
        sessions = []

        for hostname in settings.MCP_HOSTNAMES:
            # Generate unique correlation ID for each server connection
            server_correlation_id = generate_correlation_id()
            try:
                logger.info(
                    "mcp_server_connection_attempt",
                    correlation_id=server_correlation_id,
                    init_correlation_id=init_correlation_id,
                    hostname=hostname,
                )
                session = await self._exit_stack.enter_async_context(
                    mcp_sse_client(hostname, correlation_id=server_correlation_id)
                )
                session_tools = await load_mcp_tools(session)
                tools.extend(session_tools)
                sessions.append(session)
                logger.info(
                    "connected_to_mcp_server",
                    correlation_id=server_correlation_id,
                    init_correlation_id=init_correlation_id,
                    hostname=hostname,
                    tool_count=len(session_tools),
                )
            except Exception as e:
                logger.error(
                    "failed_to_connect_to_mcp_server",
                    correlation_id=server_correlation_id,
                    init_correlation_id=init_correlation_id,
                    hostname=hostname,
                    error=str(e),
                )

        self._resource = Resource(tools=tools, sessions=sessions)
        self._initialized = True
        logger.info(
            "mcp_initialization_completed",
            correlation_id=init_correlation_id,
            total_tools=len(tools),
            total_sessions=len(sessions),
        )
        return self._resource

    async def reconnect(self) -> bool:
        """Attempt to reconnect to MCP servers."""
        reconnect_correlation_id = generate_correlation_id()
        logger.info("reconnecting_to_mcp_servers", correlation_id=reconnect_correlation_id)
        try:
            await self.cleanup()
            await self.initialize()
            logger.info("mcp_reconnection_successful", correlation_id=reconnect_correlation_id)
            return True
        except Exception as e:
            logger.error("mcp_reconnection_failed", correlation_id=reconnect_correlation_id, error=str(e))
            return False

    async def cleanup(self):
        """Close all MCP sessions."""
        if self._exit_stack is not None:
            cleanup_correlation_id = generate_correlation_id()
            logger.info("mcp_cleanup_started", correlation_id=cleanup_correlation_id)
            try:
                await self._exit_stack.__aexit__(None, None, None)
                logger.info("mcp_sessions_closed_successfully", correlation_id=cleanup_correlation_id)
            except Exception as e:
                logger.error("mcp_sessions_cleanup_failed", correlation_id=cleanup_correlation_id, error=str(e))
            finally:
                self._exit_stack = None
                self._resource = None
                self._initialized = False
                logger.info("mcp_cleanup_completed", correlation_id=cleanup_correlation_id)

    def get_resource(self) -> Resource:
        """Get the current MCP resource."""
        if not self._initialized or self._resource is None:
            raise RuntimeError("MCP session manager not initialized")
        return self._resource


# Module-level singleton
_mcp_session_manager: Optional[MCPSessionManager] = None


def get_mcp_session_manager() -> MCPSessionManager:
    """Get the global MCP session manager."""
    global _mcp_session_manager
    if _mcp_session_manager is None:
        _mcp_session_manager = MCPSessionManager()
    return _mcp_session_manager
