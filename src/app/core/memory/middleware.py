"""Middleware that integrates long-term memory retrieval and update."""

from typing import Optional

from src.app.core.memory.memory import MemoryService
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult


class MemoryMiddleware(AgentMiddleware):
    """Retrieves relevant memory before invoke, updates memory after."""

    def __init__(self, memory: Optional[MemoryService] = None) -> None:
        self._memory = memory

    def _get_memory(self) -> MemoryService:
        if self._memory is None:
            from src.app.core.memory import memory_service

            self._memory = memory_service
        return self._memory

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        memory = self._get_memory()
        if ctx.messages:
            retrieved = await memory.search(ctx.user_id, ctx.messages[-1].content)
            ctx.metadata["long_term_memory"] = retrieved or "No relevant memory found."
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        memory = self._get_memory()
        if result:
            messages_dict = [dict(role=m.role, content=str(m.content)) for m in result]
            memory.schedule_add(
                ctx.user_id,
                messages_dict,
                {
                    "session_id": ctx.session_id,
                    "agent_name": ctx.agent_name,
                    "user_id": ctx.user_id,
                },
            )
        return result
