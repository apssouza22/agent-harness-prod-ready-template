from src.app.core.db.factory import make_database_cached
from src.app.core.session.factory import make_session_repository
from src.app.core.tracing.factory import (
    init_langfuse,
    make_langfuse_callback_handler,
    shutdown_langfuse,
)
from src.app.core.user.factory import make_user_repository
from src.app.core.mcp.session_manager import get_mcp_session_manager
from src.app.core.common.config import settings
from src.app.core.common.logging import logger

dbsession = make_database_cached().get_session_maker()
user_repository = make_user_repository(dbsession)
session_repository = make_session_repository(dbsession)


def langfuse_init():
    init_langfuse(settings)


def langfuse_shutdown():
    shutdown_langfuse()


async def mcp_dependencies_init():
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


async def mcp_dependencies_cleanup():
    if settings.MCP_ENABLED and settings.MCP_HOSTNAMES:
        mcp_manager = get_mcp_session_manager()
        await mcp_manager.cleanup()
        logger.info("mcp_cleaned_up")
    else:
        logger.info("mcp_cleanup_skipped")


langfuse_callback_handler = make_langfuse_callback_handler()
