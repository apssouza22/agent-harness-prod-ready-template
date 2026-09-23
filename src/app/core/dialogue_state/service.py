"""Dialogue state service facade."""

import asyncio
from typing import Any, Optional

from src.app.core.common.config import Settings, settings
from src.app.core.common.logging import logger
from src.app.core.dialogue_state.engine import DialogueStateEngine


class DialogueStateService:
    """Service for session-scoped dialogue state operations."""

    def __init__(self, app_settings: Settings | None = None) -> None:
        self._settings = app_settings or settings
        self._engine: Optional[DialogueStateEngine] = None

    async def _get_engine(self) -> DialogueStateEngine:
        if self._engine is None:
            self._engine = DialogueStateEngine(self._settings)
            await self._engine.initialize()
            logger.info("dialogue_state_initialized", table_name=self._settings.DIALOGUE_STATE_TABLE_NAME)
        return self._engine

    async def get_formatted(self, session_id: str) -> str:
        """Return formatted dialogue state for prompt injection."""
        if not self._settings.DIALOGUE_STATE_ENABLED:
            return ""

        try:
            engine = await self._get_engine()
            return await engine.get_formatted(session_id)
        except Exception as e:
            logger.exception("failed_to_get_dialogue_state", session_id=session_id, error=str(e))
            return ""

    async def update(
        self,
        session_id: str,
        user_id: int,
        messages: list[dict[str, Any]],
    ) -> None:
        """Persist an updated dialogue state for the session."""
        if not self._settings.DIALOGUE_STATE_ENABLED:
            return

        try:
            engine = await self._get_engine()
            await engine.update(session_id, str(user_id), messages)
            logger.info("dialogue_state_updated_successfully", session_id=session_id, user_id=user_id)
        except Exception as e:
            logger.exception(
                "failed_to_update_dialogue_state",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )

    def schedule_update(
        self,
        session_id: str,
        user_id: int,
        messages: list[dict[str, Any]],
    ) -> None:
        """Schedule a dialogue state update in the background."""
        if not self._settings.DIALOGUE_STATE_ENABLED:
            return

        asyncio.create_task(self.update(session_id, user_id, messages))
