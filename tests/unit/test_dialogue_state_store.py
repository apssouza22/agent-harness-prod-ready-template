"""Unit tests for DialogueStateStore."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from psycopg.types.json import Json

from src.app.core.dialogue_state.models import DialogueState
from src.app.core.dialogue_state.store import DialogueStateStore


@pytest.fixture
def store() -> DialogueStateStore:
    from src.app.core.common.config import settings

    return DialogueStateStore(settings)


@pytest.mark.asyncio
async def test_upsert_wraps_state_payload_as_json(store: DialogueStateStore) -> None:
    mock_cursor = AsyncMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_pool = MagicMock()
    mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)

    state = DialogueState(topic="travel planning", slots={"destination": "Lisbon"})
    store._initialized = True

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(store, "_get_pool", AsyncMock(return_value=mock_pool))
        await store.upsert("session-1", "42", state)

    mock_cursor.execute.assert_awaited_once()
    execute_args = mock_cursor.execute.await_args.args
    query_params = execute_args[1]
    assert query_params[0] == "session-1"
    assert query_params[1] == "42"
    assert isinstance(query_params[2], Json)
    assert query_params[2].obj == state.model_dump()
