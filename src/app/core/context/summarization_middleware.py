"""Middleware that summarizes conversation history to prevent context overflow.

Hooks into ``before_model_call`` so that every LLM invocation inside the
graph automatically receives a context-budget-safe message list.  The
actual summarization logic is delegated to
:func:`src.app.core.context.summarize_if_too_long`.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from src.app.core.context.summarizer import summarize_if_too_long
from src.app.core.middleware.types import AgentContext, AgentMiddleware


class SummarizationMiddleware(AgentMiddleware):
    """Compresses conversation history when it approaches the model's context limit.

    The middleware is a no-op when the conversation still fits within budget,
    so it is safe to keep it permanently in the pipeline.

    Args:
        llm: Chat model instance used to generate summaries when needed.
        model_name: Model identifier for context-window lookup
            (e.g. ``"openai:gpt-5-mini"``).
    """

    def __init__(self, llm: BaseChatModel, model_name: str) -> None:
        self._llm = llm
        self._model_name = model_name

    async def before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        return await summarize_if_too_long(
            messages=messages,
            model_name=self._model_name,
            llm=self._llm,
            session_id=ctx.session_id,
        )
