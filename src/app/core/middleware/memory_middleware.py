"""Middleware that integrates long-term memory retrieval and update."""

from typing import Optional

from src.app.core.memory import MemoryService, memory_service
from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult


class MemoryMiddleware(AgentMiddleware):
    """Retrieves relevant memory before invoke, updates memory after."""

    def __init__(self, memory: Optional[MemoryService] = None) -> None:
        self._memory = memory or memory_service

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        if ctx.messages:
            memory = await self._memory.search(ctx.user_id, ctx.messages[-1].content)
            ctx.metadata["long_term_memory"] = memory or "No relevant memory found."
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        if result:
            messages_dict = [dict(role=m.role, content=str(m.content)) for m in result]
            self._memory.schedule_add(
                ctx.user_id,
                messages_dict,
                {
                    "session_id": ctx.session_id,
                    "agent_name": ctx.agent_name,
                    "user_id": ctx.user_id,
                },
            )
        return result
