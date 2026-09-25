"""Middleware that integrates dialogue state tracking into the agent pipeline."""

from src.app.core.common.config import settings
from src.app.core.dialogue_state.service import DialogueStateService
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult


class DialogueStateMiddleware(AgentMiddleware):
    """Loads dialogue state before invoke and updates it after."""

    def __init__(self, dialogue_state: DialogueStateService) -> None:
        self._dialogue_state = dialogue_state

    async def before_invoke(self, ctx: AgentContext) -> InvokeResult | None:
        if not settings.DIALOGUE_STATE_ENABLED:
            return None

        formatted_state = await self._dialogue_state.get_formatted(ctx.session_id)
        ctx.metadata["dialogue_state"] = formatted_state or "No structured dialogue state yet."
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        if not settings.DIALOGUE_STATE_ENABLED:
            return result

        if result:
            messages_dict = [dict(role=m.role, content=str(m.content)) for m in result]
            self._dialogue_state.schedule_update(ctx.session_id, ctx.user_id, messages_dict)
        return result
