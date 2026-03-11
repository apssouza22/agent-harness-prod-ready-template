"""Context management module for LLM context overflow prevention.

Provides utilities to:
- Evict oversized tool results to disk with head/tail previews.
- Summarize conversation history when it approaches the model's context
  window, offloading old messages to a markdown file.

Usage:
    from src.app.core.context import truncate_tool_call_if_too_long, summarize_if_too_long

    tool_message = truncate_tool_call_if_too_long(tool_message)
    messages = await summarize_if_too_long(messages, model_name, llm, session_id)
"""

from src.app.core.context.context_manager import truncate_tool_call_if_too_long
from src.app.core.context.summarizer import summarize_if_too_long
