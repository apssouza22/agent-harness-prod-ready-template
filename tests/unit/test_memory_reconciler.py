"""Unit tests for memory reconciliation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.common.config import settings
from src.app.core.memory.models import MemoryReconcileItem, MemoryReconcileOutput
from src.app.core.memory.reconciler import MemoryReconciler


@pytest.fixture
def reconciler() -> MemoryReconciler:
    return MemoryReconciler(settings, chat_model=MagicMock())


def test_parse_actions_maps_update_delete_and_add(reconciler: MemoryReconciler) -> None:
    payload = MemoryReconcileOutput(
        memory=[
            MemoryReconcileItem(
                id="0",
                text="Works at Acme Corp",
                event="UPDATE",
                old_memory="Works at Old Corp",
            ),
            MemoryReconcileItem(id="1", text="Loves cheese pizza", event="DELETE"),
            MemoryReconcileItem(id="2", text="Prefers dark mode", event="NONE"),
            MemoryReconcileItem(id="3", text="Uses Python daily", event="ADD"),
        ]
    )
    id_mapping = {"0": "uuid-0", "1": "uuid-1", "2": "uuid-2"}

    actions = reconciler._parse_actions(payload, id_mapping)

    assert len(actions) == 4
    assert actions[0].event == "UPDATE"
    assert actions[0].memory_id == "uuid-0"
    assert actions[0].previous_memory == "Works at Old Corp"
    assert actions[1].event == "DELETE"
    assert actions[1].memory_id == "uuid-1"
    assert actions[2].event == "NONE"
    assert actions[2].memory_id == "uuid-2"
    assert actions[3].event == "ADD"
    assert actions[3].memory_id is None


def test_parse_actions_skips_unknown_ids(reconciler: MemoryReconciler) -> None:
    payload = MemoryReconcileOutput(
        memory=[MemoryReconcileItem(id="99", text="Unknown", event="DELETE")]
    )

    actions = reconciler._parse_actions(payload, {"0": "uuid-0"})

    assert actions == []


@pytest.mark.asyncio
async def test_reconcile_without_existing_memories_adds_all_facts(reconciler: MemoryReconciler) -> None:
    actions = await reconciler.reconcile([], ["Name is John", "Uses Rust"], id_mapping={})

    assert [action.event for action in actions] == ["ADD", "ADD"]
    assert [action.text for action in actions] == ["Name is John", "Uses Rust"]


@pytest.mark.asyncio
async def test_reconcile_uses_injected_chat_model() -> None:
    mock_wrapped = AsyncMock()
    mock_wrapped.ainvoke.return_value = MemoryReconcileOutput(
        memory=[
            MemoryReconcileItem(
                id="0",
                text="Works at Acme Corp",
                event="UPDATE",
                old_memory="Works at Old Corp",
            ),
        ]
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_wrapped
    reconciler = MemoryReconciler(settings, chat_model=mock_llm)

    actions = await reconciler.reconcile(
        [{"id": "0", "text": "Works at Old Corp"}],
        ["Works at Acme Corp"],
        id_mapping={"0": "uuid-0"},
    )

    mock_llm.with_structured_output.assert_called_once()
    mock_wrapped.ainvoke.assert_awaited_once()
    assert len(actions) == 1
    assert actions[0].event == "UPDATE"
