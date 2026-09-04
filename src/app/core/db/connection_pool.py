
from src.app.core.checkpoint.factory import make_connection_pool

__all__ = ["get_connection_pool", "make_connection_pool"]


async def get_connection_pool():
    """Backward-compatible wrapper around make_connection_pool."""
    return await make_connection_pool()
