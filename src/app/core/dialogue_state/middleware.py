"""Middleware that integrates dialogue state tracking into the agent pipeline."""

from typing import Optional

from src.app.core.common.config import settings
from src.app.core.dialogue_state.service import DialogueStateService
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult


class DialogueStateMiddleware(AgentMiddleware):
    """Loads dialogue state before invoke and updates it after."""

    def __init__(self, dialogue_state: Optional[DialogueStateService] = None) -> None:
        self._dialogue_state = dialogue_state

    def _get_service(self) -> DialogueStateService:
        if self._dialogue_state is None:
            from src.app.core.dialogue_state import dialogue_state_service

            self._dialogue_state = dialogue_state_service
        return self._dialogue_state

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        if not settings.DIALOGUE_STATE_ENABLED:
            return None

        service = self._get_service()
        formatted_state = await service.get_formatted(ctx.session_id)
        ctx.metadata["dialogue_state"] = formatted_state or "No structured dialogue state yet."
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        if not settings.DIALOGUE_STATE_ENABLED:
            return result

        service = self._get_service()
        if result:
            messages_dict = [dict(role=m.role, content=str(m.content)) for m in result]
            service.schedule_update(ctx.session_id, ctx.user_id, messages_dict)
        return result
