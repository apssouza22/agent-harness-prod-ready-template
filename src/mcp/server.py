from mcp.server.mcpserver import MCPServer

from src.app.core.common.config import settings

mcp_server = MCPServer("MCP Server")


@mcp_server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp_server.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    # Streamable HTTP is the modern default; use transport="sse" for legacy servers.
    mcp_server.run(
        transport="streamable-http",
        port=settings.MCP_SERVER_PORT,
    )
