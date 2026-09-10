"""Shared HTTP connection pool for MCP clients in long-lived deployments."""

from typing import Any

import httpx2

_POOL = httpx2.AsyncHTTPTransport()


class _SharedPool(httpx2.AsyncBaseTransport):
    """Lend the shared transport to each client without letting any client close it."""

    handle_async_request = _POOL.handle_async_request

    async def aclose(self) -> None:
        return None


def mcp_httpx_client_factory(**kwargs: Any) -> httpx2.AsyncClient:
    """Create httpx clients that share one connection pool across MCP servers."""
    return httpx2.AsyncClient(transport=_SharedPool(), **kwargs)


async def close_mcp_connection_pool() -> None:
    """Close the shared MCP HTTP connection pool on application shutdown."""
    await _POOL.aclose()
