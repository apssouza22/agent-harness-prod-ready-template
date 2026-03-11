"""Context management for LLM tool results to prevent context overflow.

Large tool results are evicted to the filesystem and replaced with a
head/tail preview so the LLM stays within its context window.

When a ToolMessage exceeds the character threshold (~80k chars ≈ 20k tokens),
the full content is persisted to disk and the in-state message is replaced
with a compact preview plus a file path the LLM can read_file on later.
"""

import os
from pathlib import Path

from langchain_core.messages import ToolMessage

from src.app.core.common.logging import logger

MAX_TOOL_RESULT_CHARS = 80_000
PREVIEW_LINES = 5
MAX_LINE_CHARS = 1_000

LARGE_RESULTS_DIR = Path(os.getenv("LARGE_TOOL_RESULTS_DIR", "large_tool_results"))


def truncate_tool_call_if_too_long(tool_message: ToolMessage) -> ToolMessage:
    """Evict oversized tool results to disk and replace with a preview.

    If the tool message content exceeds MAX_TOOL_RESULT_CHARS (~20k tokens),
    the full text is written to LARGE_RESULTS_DIR/{tool_call_id} and the
    message content is replaced with a head/tail preview plus a file path.

    Args:
        tool_message: The ToolMessage to check and potentially evict.

    Returns:
        The original message if small enough, or a new message with preview content.
    """
    content = tool_message.content
    if not isinstance(content, str) or len(content) <= MAX_TOOL_RESULT_CHARS:
        return tool_message

    file_path = _write_large_result(tool_message.tool_call_id, content)
    preview = _build_preview(content)

    evicted_content = (
        f"[Tool result too large ({len(content):,} chars). "
        f"Full output saved to: {file_path}]\n\n"
        f"--- Preview (first {PREVIEW_LINES} lines) ---\n"
        f"{preview['head']}\n"
        f"...\n"
        f"--- Preview (last {PREVIEW_LINES} lines) ---\n"
        f"{preview['tail']}\n\n"
        f'Use read_file("{file_path}") to access the full content.'
    )

    logger.info(
        "tool_result_evicted",
        tool_call_id=tool_message.tool_call_id,
        tool_name=tool_message.name,
        original_chars=len(content),
        preview_chars=len(evicted_content),
        file_path=str(file_path),
    )

    return ToolMessage(
        content=evicted_content,
        name=tool_message.name,
        tool_call_id=tool_message.tool_call_id,
    )


def _build_preview(content: str) -> dict[str, str]:
    """Build head/tail preview from content.

    Extracts the first and last PREVIEW_LINES lines, each capped at
    MAX_LINE_CHARS characters.

    Args:
        content: The full text content.

    Returns:
        Dict with 'head' and 'tail' preview strings.
    """
    lines = content.splitlines()

    head_lines = [line[:MAX_LINE_CHARS] for line in lines[:PREVIEW_LINES]]
    tail_lines = [line[:MAX_LINE_CHARS] for line in lines[-PREVIEW_LINES:]]

    return {
        "head": "\n".join(head_lines),
        "tail": "\n".join(tail_lines),
    }


def _write_large_result(tool_call_id: str, content: str) -> Path:
    """Write large tool result to filesystem.

    Args:
        tool_call_id: Unique identifier for the tool call.
        content: The full text content to write.

    Returns:
        Path to the written file.
    """
    LARGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_call_id)
    file_path = LARGE_RESULTS_DIR / safe_id
    file_path.write_text(content, encoding="utf-8")

    logger.debug(
        "large_tool_result_written",
        tool_call_id=tool_call_id,
        file_path=str(file_path),
        content_length=len(content),
    )

    return file_path
