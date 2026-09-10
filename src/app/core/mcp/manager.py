"""Production MCP management using LangChain MCPAdapter and FastMCP ClientGroup."""

import uuid
from contextlib import AsyncExitStack
from typing import Optional

from fastmcp.client import Client
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain_core.tools import BaseTool
from mcp.client.caching import CacheMode
from pydantic import BaseModel, ConfigDict

from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.mcp.connection_pool import close_mcp_connection_pool, mcp_httpx_client_factory
from src.app.core.mcp.url_utils import normalize_mcp_server_url, server_name_from_url


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracking MCP operations."""
    return str(uuid.uuid4())


class Resource(BaseModel):
    """Discovered MCP tools and connected server metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[BaseTool]
    server_count: int = 0


def _build_client(server_url: str) -> Client:
    """Create a FastMCP client for one MCP server URL."""
    return Client(
        StreamableHttpTransport(
            server_url,
            httpx_client_factory=mcp_httpx_client_factory,
        ),
        mode=settings.MCP_PROTOCOL_MODE,
    )


class McpManager:
    """Manages MCP clients and discovered tools for the application lifetime."""

    def __init__(self) -> None:
        self._exit_stack: Optional[AsyncExitStack] = None
        self._resource: Optional[Resource] = None

    async def initialize(self, *, cache_mode: CacheMode | None = None) -> Resource:
        """Initialize MCP clients, discover tools, and keep connections open."""
        if self._resource is not None:
            return self._resource

        init_correlation_id = generate_correlation_id()
        logger.info("mcp_initialization_started", correlation_id=init_correlation_id)

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        connected_clients: dict[str, Client] = {}
        discovery_mode = cache_mode or settings.MCP_TOOL_CACHE_MODE

        for index, hostname in enumerate(settings.MCP_HOSTNAMES):
            server_correlation_id = generate_correlation_id()
            server_url = normalize_mcp_server_url(
                hostname,
                default_path=settings.MCP_ENDPOINT_PATH,
            )
            server_name = server_name_from_url(server_url, index)

            try:
                logger.info(
                    "mcp_server_connection_attempt",
                    correlation_id=server_correlation_id,
                    init_correlation_id=init_correlation_id,
                    hostname=hostname,
                    server_url=server_url,
                    server_name=server_name,
                    protocol_mode=settings.MCP_PROTOCOL_MODE,
                )
                client = _build_client(server_url)
                await self._exit_stack.enter_async_context(client)
                connected_clients[server_name] = client
                logger.info(
                    "connected_to_mcp_server",
                    correlation_id=server_correlation_id,
                    init_correlation_id=init_correlation_id,
                    server_name=server_name,
                    server_url=server_url,
                )
            except Exception:
                logger.exception(
                    "failed_to_connect_to_mcp_server",
                    correlation_id=server_correlation_id,
                    init_correlation_id=init_correlation_id,
                    hostname=hostname,
                    server_url=server_url,
                )

        tools: list[BaseTool] = []
        if connected_clients:
            client_group = ClientGroup(connected_clients)
            adapter = MCPAdapter(client_group)
            tools = await adapter.list_tools(cache_mode=discovery_mode)
        else:
            logger.warning(
                "mcp_no_servers_connected",
                correlation_id=init_correlation_id,
            )

        self._resource = Resource(tools=tools, server_count=len(connected_clients))
        logger.info(
            "mcp_initialization_completed",
            correlation_id=init_correlation_id,
            total_tools=len(tools),
            total_servers=len(connected_clients),
            cache_mode=discovery_mode,
        )
        return self._resource

    async def reconnect(self) -> bool:
        """Reconnect to MCP servers and refresh the tool catalog."""
        reconnect_correlation_id = generate_correlation_id()
        logger.info("reconnecting_to_mcp_servers", correlation_id=reconnect_correlation_id)
        try:
            await self.cleanup()
            await self.initialize(cache_mode="refresh")
            logger.info("mcp_reconnection_successful", correlation_id=reconnect_correlation_id)
            return True
        except Exception:
            logger.exception(
                "mcp_reconnection_failed",
                correlation_id=reconnect_correlation_id,
            )
            return False

    async def cleanup(self) -> None:
        """Close MCP clients and the shared HTTP connection pool."""
        if self._exit_stack is None:
            return

        cleanup_correlation_id = generate_correlation_id()
        logger.info("mcp_cleanup_started", correlation_id=cleanup_correlation_id)
        try:
            await self._exit_stack.__aexit__(None, None, None)
            logger.info("mcp_sessions_closed_successfully", correlation_id=cleanup_correlation_id)
        except Exception:
            logger.exception(
                "mcp_sessions_cleanup_failed",
                correlation_id=cleanup_correlation_id,
            )
        finally:
            self._exit_stack = None
            self._resource = None
            await close_mcp_connection_pool()
            logger.info("mcp_cleanup_completed", correlation_id=cleanup_correlation_id)

    def get_resource(self) -> Resource:
        """Get the current MCP resource."""
        if self._resource is None:
            raise RuntimeError("MCP manager not initialized")
        return self._resource
