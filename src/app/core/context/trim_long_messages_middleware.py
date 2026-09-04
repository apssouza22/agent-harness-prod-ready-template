"""Middleware that trims conversation messages to fit within the model's token budget.

Hooks into ``before_model_call`` so that every LLM invocation inside the
graph receives a message list that respects the configured token limit.
Uses LangChain's ``trim_messages`` with a *last* strategy to keep the
most recent messages when the conversation exceeds the budget.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import trim_messages

from src.app.core.common.logging import logger
from src.app.core.middleware.types import AgentContext, AgentMiddleware


class TrimLongMessagesMiddleware(AgentMiddleware):
    """Trims conversation messages to stay within the model's token limit.

    The middleware is a no-op when the conversation already fits within
    budget, so it is safe to keep permanently in the pipeline.

    Args:
        llm: Chat model instance used for token counting.
        max_tokens: Maximum token budget for the message list.
    """

    def __init__(self, llm: BaseChatModel, max_tokens: int) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    async def before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        try:
            return trim_messages(
                messages,
                strategy="last",
                token_counter=self._llm,
                max_tokens=self._max_tokens,
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
        except ValueError as e:
            if "Unrecognized content block type" in str(e):
                logger.warning(
                    "token_counting_failed_skipping_trim",
                    error=str(e),
                    message_count=len(messages),
                    session_id=ctx.session_id,
                )
                return messages
            raise
