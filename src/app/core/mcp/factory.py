"""MCP manager factory."""

from src.app.core.mcp.manager import McpManager


def make_mcp_manager() -> McpManager:
    """Create an MCP manager for application lifecycle wiring."""
    return McpManager()
