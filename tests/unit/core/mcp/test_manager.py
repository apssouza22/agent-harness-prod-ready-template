"""Unit tests for MCP manager initialization."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.tools import StructuredTool

from src.app.core.mcp.manager import McpManager


def _forecast_tool(query: str) -> str:
    """Return the weather forecast query."""
    return query


@pytest.mark.asyncio
async def test_initialize_returns_empty_resource_when_no_servers_connect() -> None:
    manager = McpManager()

    with patch("src.app.core.mcp.manager.settings") as mock_settings:
        mock_settings.MCP_HOSTNAMES = ["http://bad-host:7001"]
        mock_settings.MCP_ENDPOINT_PATH = "/mcp"
        mock_settings.MCP_PROTOCOL_MODE = "auto"
        mock_settings.MCP_TOOL_CACHE_MODE = "use"

        with patch("src.app.core.mcp.manager.Client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.side_effect = ConnectionError("unreachable")
            mock_client_cls.return_value = mock_client

            resource = await manager.initialize()

    assert resource.tools == []
    assert resource.server_count == 0
    await manager.cleanup()


@pytest.mark.asyncio
async def test_initialize_discovers_tools_from_connected_servers() -> None:
    manager = McpManager()
    mock_tool = StructuredTool.from_function(
        func=_forecast_tool,
        name="weather_get_forecast",
    )

    with patch("src.app.core.mcp.manager.settings") as mock_settings:
        mock_settings.MCP_HOSTNAMES = ["http://localhost:7001"]
        mock_settings.MCP_ENDPOINT_PATH = "/mcp"
        mock_settings.MCP_PROTOCOL_MODE = "auto"
        mock_settings.MCP_TOOL_CACHE_MODE = "use"

        with (
            patch("src.app.core.mcp.manager.Client") as mock_client_cls,
            patch("src.app.core.mcp.manager.ClientGroup") as mock_group_cls,
            patch("src.app.core.mcp.manager.MCPAdapter") as mock_adapter_cls,
        ):
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            mock_group = AsyncMock()
            mock_group_cls.return_value = mock_group

            mock_adapter = MagicMock()
            mock_adapter.list_tools = AsyncMock(return_value=[mock_tool])
            mock_adapter_cls.return_value = mock_adapter

            resource = await manager.initialize()

    assert resource.server_count == 1
    assert len(resource.tools) == 1
    mock_adapter.list_tools.assert_awaited_once_with(cache_mode="use")
    await manager.cleanup()
