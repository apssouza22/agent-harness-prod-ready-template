"""Conversation history summarization to prevent LLM context overflow.

When the conversation reaches 85 % of the model's context window the
pipeline kicks in with two stages:

1. **Lightweight truncation** – tool-call arguments (e.g. file contents
   passed to ``write_file``) in *older* messages are trimmed to 2 K chars.
   If that alone brings the token count below the threshold, no LLM call
   is needed.

2. **Full summarization** – the older portion of the conversation is
   summarised by the LLM and the original messages are offloaded to a
   markdown file on disk.  The summary is injected as a
   ``SystemMessage`` so subsequent turns retain the essential context.

Token counting uses ``count_tokens_approximately`` from langchain-core
for the threshold check.  The cheaper char-based heuristic
(``NUM_CHARS_PER_TOKEN = 4``) is used for the split-point search and
argument truncation to avoid repeated full counts.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately

from src.app.core.common.logging import logger
from src.app.core.common.token_limit import get_model_token_limit

NUM_CHARS_PER_TOKEN = 4
CONTEXT_THRESHOLD_RATIO = 0.85
RECENT_CONTEXT_RATIO = 0.30
TOOL_ARG_TRUNCATE_CHARS = 2_000
DEFAULT_TOKEN_LIMIT = 128_000
MAX_SUMMARY_INPUT_CHARS = 200_000
MIN_MESSAGES_TO_SUMMARIZE = 6

SUMMARIES_DIR = Path(os.getenv("CONVERSATION_SUMMARIES_DIR", "conversation_summaries"))

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Produce a concise briefing that "
    "preserves key facts, user requests, decisions, file paths, technical "
    "details, tool results, and current task status. Omit pleasantries, "
    "repetitive content, and fully resolved issues."
)


async def summarize_if_too_long(
    messages: list[BaseMessage],
    model_name: str,
    llm: BaseChatModel,
    session_id: str = "unknown",
) -> list[BaseMessage]:
    """Summarize conversation history when it approaches the model's context limit.

    The function is safe to call on every turn – it is a no-op when the
    context is still within budget.

    Args:
        messages: Full conversation history (LangGraph state messages).
        model_name: Model identifier used to look up the context window
            size (e.g. ``"openai:gpt-5-mini"``).
        llm: Chat model instance used to generate the summary.
        session_id: Session identifier used for the offloaded markdown
            file name.

    Returns:
        The (possibly compressed) message list.
    """
    if len(messages) < MIN_MESSAGES_TO_SUMMARIZE:
        return messages

    token_limit = get_model_token_limit(model_name) or DEFAULT_TOKEN_LIMIT
    threshold = int(token_limit * CONTEXT_THRESHOLD_RATIO)

    current_tokens = count_tokens_approximately(messages)
    if current_tokens < threshold:
        return messages

    logger.info(
        "context_summarization_triggered",
        current_tokens=current_tokens,
        threshold=threshold,
        token_limit=token_limit,
        message_count=len(messages),
        session_id=session_id,
    )

    split_idx = _find_safe_split_index(messages, token_limit)
    if split_idx == 0:
        return messages

    old_messages = messages[:split_idx]
    recent_messages = messages[split_idx:]

    # Stage 1 – truncate bulky tool-call arguments in old messages
    truncated_old = _truncate_tool_call_args(old_messages)
    candidate = truncated_old + recent_messages

    tokens_after_truncation = count_tokens_approximately(candidate)
    if tokens_after_truncation < threshold:
        logger.info(
            "context_reduced_by_arg_truncation",
            tokens_before=current_tokens,
            tokens_after=tokens_after_truncation,
            threshold=threshold,
            session_id=session_id,
        )
        return candidate

    # Stage 2 – full LLM summarization + offload to markdown
    file_path = _write_messages_to_markdown(old_messages, session_id)
    summary_text = await _generate_summary(truncated_old, llm)

    summary_message = HumanMessage(
        content=(
            f"[Summary of earlier conversation ({len(old_messages)} messages). "
            f"Full transcript: {file_path}]\n\n{summary_text}"
        )
    )

    result = [summary_message] + recent_messages

    logger.info(
        "context_summarized",
        old_message_count=len(old_messages),
        recent_message_count=len(recent_messages),
        tokens_before=current_tokens,
        tokens_after=count_tokens_approximately(result),
        summary_file=str(file_path),
        session_id=session_id,
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_safe_split_index(messages: list[BaseMessage], token_limit: int) -> int:
    """Find a split point that keeps ~30 % of the context for recent messages.

    The split is snapped to the nearest ``HumanMessage`` boundary so that
    tool-call chains (``AIMessage`` with calls → ``ToolMessage`` responses)
    are never broken across the boundary.

    Returns 0 when no valid split exists (everything is recent).
    """
    recent_chars_budget = int(token_limit * RECENT_CONTEXT_RATIO * NUM_CHARS_PER_TOKEN)
    chars_from_end = 0
    raw_split = 0

    for i in range(len(messages) - 1, -1, -1):
        chars_from_end += _estimate_message_chars(messages[i])
        if chars_from_end >= recent_chars_budget:
            raw_split = i + 1
            break

    if raw_split <= 0:
        return 0

    # Snap forward to the next HumanMessage
    for j in range(raw_split, len(messages)):
        if isinstance(messages[j], HumanMessage):
            return j

    # Snap backward to the previous HumanMessage
    for j in range(raw_split - 1, 0, -1):
        if isinstance(messages[j], HumanMessage):
            return j

    return 0


def _estimate_message_chars(message: BaseMessage) -> int:
    """Quick char-count estimate for a message including tool-call payloads."""
    content = message.content
    chars = len(content) if isinstance(content, str) else len(str(content))
    if hasattr(message, "tool_calls") and message.tool_calls:
        chars += sum(
            len(json.dumps(tc.get("args", {})))
            for tc in message.tool_calls
        )
    return chars


def _truncate_tool_call_args(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Truncate tool-call argument values longer than TOOL_ARG_TRUNCATE_CHARS.

    Only ``AIMessage`` instances with ``tool_calls`` are affected.
    Returns new message objects; the originals are not mutated.
    """
    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            truncated_calls = []
            any_truncated = False
            for tc in msg.tool_calls:
                new_args = {}
                for key, value in tc.get("args", {}).items():
                    if isinstance(value, str) and len(value) > TOOL_ARG_TRUNCATE_CHARS:
                        new_args[key] = (
                            value[:TOOL_ARG_TRUNCATE_CHARS]
                            + f"\n... [truncated, {len(value):,} chars total]"
                        )
                        any_truncated = True
                    else:
                        new_args[key] = value
                truncated_calls.append({**tc, "args": new_args})
            if any_truncated:
                result.append(msg.model_copy(update={"tool_calls": truncated_calls}))
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


async def _generate_summary(messages: list[BaseMessage], llm: BaseChatModel) -> str:
    """Ask the LLM to summarize the old conversation segment."""
    conversation_text = _format_messages_as_text(messages)

    if len(conversation_text) > MAX_SUMMARY_INPUT_CHARS:
        half = MAX_SUMMARY_INPUT_CHARS // 2
        conversation_text = (
            conversation_text[:half]
            + "\n\n... [middle section omitted for brevity] ...\n\n"
            + conversation_text[-half:]
        )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Summarize this conversation:\n\n{conversation_text}"
            ),
        ])
        return str(response.content)
    except Exception as e:
        logger.warning("summarization_llm_call_failed", error=str(e), exc_info=True)
        fallback_lines = []
        for msg in messages:
            role = type(msg).__name__
            content = str(msg.content)[:200]
            fallback_lines.append(f"- {role}: {content}...")
        return "Summarization failed. Message overview:\n" + "\n".join(fallback_lines)


def _format_messages_as_text(messages: list[BaseMessage]) -> str:
    """Render messages into a plain-text transcript for the summarizer."""
    lines: list[str] = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if len(content) > 4_000:
            content = content[:2_000] + "\n...[truncated]...\n" + content[-2_000:]
        lines.append(f"[{role}]: {content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc.get("args", {}))
                if len(args_str) > 500:
                    args_str = args_str[:500] + "..."
                lines.append(f"  -> Tool call: {tc['name']}({args_str})")
    return "\n\n".join(lines)


def _write_messages_to_markdown(
    messages: list[BaseMessage], session_id: str
) -> Path:
    """Persist the full old-message segment to a markdown file."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in session_id
    )
    file_path = SUMMARIES_DIR / f"{safe_session}_{timestamp}.md"

    lines: list[str] = [
        f"# Conversation History – Session {session_id}\n\n",
        f"Offloaded at: {datetime.now().isoformat()}\n",
        f"Messages: {len(messages)}\n\n---\n\n",
    ]

    for i, msg in enumerate(messages):
        role = type(msg).__name__.replace("Message", "").upper()
        lines.append(f"## Message {i + 1} ({role})\n\n")

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{content}\n\n")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            lines.append("**Tool Calls:**\n\n")
            for tc in msg.tool_calls:
                lines.append(f"- `{tc['name']}` (id: {tc.get('id', 'n/a')})\n")
                lines.append(
                    f"  ```json\n  {json.dumps(tc.get('args', {}), indent=2)}\n  ```\n\n"
                )

    file_path.write_text("".join(lines), encoding="utf-8")

    logger.debug(
        "conversation_offloaded_to_markdown",
        file_path=str(file_path),
        message_count=len(messages),
        session_id=session_id,
    )

    return file_path
