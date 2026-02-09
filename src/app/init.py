import os

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.db.database import database_factory
from src.app.core.mcp.session_manager import get_mcp_session_manager
from src.app.core.session import SessionRepository
from src.app.core.user import UserRepository

dbsession = database_factory.get_session_maker()
user_repository =  UserRepository(dbsession)
session_repository = SessionRepository(dbsession)


def langfuse_init():
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    if langfuse.auth_check():
        logger.info("langfuse_auth_success")
    else:
        logger.error("langfuse_auth_failure")


def get_langfuse_callback_handler() -> CallbackHandler:
    """Create a Langfuse CallbackHandler for tracking LLM interactions.

    Returns:
        CallbackHandler: Configured Langfuse callback handler.
    """

    return CallbackHandler()


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


langfuse_callback_handler = get_langfuse_callback_handler()