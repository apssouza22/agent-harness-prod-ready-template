"""MCP lifecycle dependency helpers."""

from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.mcp.manager import McpManager


async def initialize_mcp_manager(mcp_manager: McpManager) -> None:
    if settings.MCP_ENABLED and settings.MCP_HOSTNAMES:
        try:
            resource = await mcp_manager.initialize()
            logger.info(
                "mcp_initialized",
                tool_count=len(resource.tools),
                server_count=resource.server_count,
            )
        except Exception:
            logger.exception("mcp_initialization_failed")
            logger.warning("continuing_without_mcp_tools")
    else:
        logger.info("mcp_disabled_or_no_hosts_configured")


async def cleanup_mcp_manager(mcp_manager: McpManager) -> None:
    if settings.MCP_ENABLED and settings.MCP_HOSTNAMES:
        await mcp_manager.cleanup()
        logger.info("mcp_cleaned_up")
    else:
        logger.info("mcp_cleanup_skipped")
