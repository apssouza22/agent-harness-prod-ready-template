"""Backward-compatible checkpoint helpers.

Prefer CheckpointService via make_checkpoint_service() or CheckpointServiceDep
for new code.
"""

from src.app.core.checkpoint.factory import make_checkpoint_service, make_checkpointer
from src.app.core.db.connection_pool import get_connection_pool


async def get_checkpointer():
    """Backward-compatible wrapper around make_checkpointer."""
    return await make_checkpointer()


async def clear_checkpoints(session_id: str) -> None:
    """Backward-compatible wrapper around CheckpointService.clear_session."""
    connection_pool = await get_connection_pool()
    service = make_checkpoint_service(connection_pool=connection_pool)
    await service.clear_session(session_id)
