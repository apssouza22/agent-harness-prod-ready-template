"""Application bootstrap helpers for MCP lifecycle."""

from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.mcp.session_manager import get_mcp_session_manager


async def mcp_dependencies_init() -> None:
    if settings.MCP_ENABLED and settings.MCP_HOSTNAMES:
        mcp_manager = get_mcp_session_manager()
        try:
            resource = await mcp_manager.initialize()
            logger.info("mcp_initialized", tool_count=len(resource.tools), session_count=len(resource.sessions))
        except Exception as e:
            logger.error("mcp_initialization_failed", error=str(e))
            logger.warning("continuing_without_mcp_tools")
    else:
        logger.info("mcp_disabled_or_no_hosts_configured")


async def mcp_dependencies_cleanup() -> None:
    if settings.MCP_ENABLED and settings.MCP_HOSTNAMES:
        mcp_manager = get_mcp_session_manager()
        await mcp_manager.cleanup()
        logger.info("mcp_cleaned_up")
    else:
        logger.info("mcp_cleanup_skipped")
