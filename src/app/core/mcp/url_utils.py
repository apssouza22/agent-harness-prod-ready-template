"""Helpers for normalizing MCP server URLs from configuration."""

from urllib.parse import urlparse, urlunparse


def normalize_mcp_server_url(hostname: str, default_path: str = "/mcp") -> str:
    """Normalize a configured MCP host into an http(s) URL FastMCP can connect to.

    Args:
        hostname: Raw host value from configuration (with or without scheme/path).
        default_path: Path appended when none is provided. Defaults to ``/mcp`` for
            streamable HTTP servers. Use ``/sse`` for legacy SSE servers.

    Returns:
        A normalized URL suitable for ``fastmcp.Client`` or ``StreamableHttpTransport``.
    """
    value = hostname.strip()
    if not value:
        raise ValueError("mcp hostname must not be empty")

    if "://" not in value:
        value = f"http://{value}"

    parsed = urlparse(value)
    if parsed.path and parsed.path != "/":
        return value.rstrip("/")

    normalized = parsed._replace(path=default_path)
    return urlunparse(normalized).rstrip("/")


def server_name_from_url(url: str, index: int) -> str:
    """Derive a stable ClientGroup key from a server URL."""
    parsed = urlparse(url)
    if parsed.netloc:
        return parsed.netloc.replace(":", "_").replace(".", "_")
    return f"server_{index}"
