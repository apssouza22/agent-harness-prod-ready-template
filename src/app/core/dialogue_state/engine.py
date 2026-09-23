"""Dialogue state engine for load, format, and update operations."""

from typing import Any

from src.app.core.common.config import Settings
from src.app.core.dialogue_state.models import DialogueState
from src.app.core.dialogue_state.store import DialogueStateStore
from src.app.core.dialogue_state.updater import DialogueStateUpdater


class DialogueStateEngine:
    """Orchestrate dialogue state persistence and LLM updates."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._store = DialogueStateStore(app_settings)
        self._updater = DialogueStateUpdater(app_settings)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize backing storage once per process."""
        if self._initialized:
            return
        await self._store.initialize()
        self._initialized = True

    async def get_formatted(self, session_id: str) -> str:
        """Load and format dialogue state for prompt injection."""
        await self.initialize()
        state = await self._store.get(session_id)
        if state is None:
            return DialogueState().to_prompt_text()
        return state.to_prompt_text()

    async def update(
        self,
        session_id: str,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Merge the latest conversation turn into persisted dialogue state."""
        await self.initialize()
        previous_state = await self._store.get(session_id) or DialogueState()
        updated_state = await self._updater.update_state(previous_state, messages)
        await self._store.upsert(session_id, user_id, updated_state)
