"""Graph node functions and message processing utilities.

This module provides factory functions for creating graph nodes and utilities
for processing messages in the LangGraph workflow.
"""

from langchain_core.messages import BaseMessage, convert_to_openai_messages
from src.app.core.common.model.message import Message


def process_messages(messages: list[BaseMessage]) -> list[Message]:
    """Convert BaseMessages to API Message format.

    Args:
        messages: List of BaseMessage objects from the graph.

    Returns:
        list[Message]: List of Message objects for the API response.
    """
    openai_style_messages = convert_to_openai_messages(messages)
    # keep just assistant and user messages
    return [
        Message(role=message["role"], content=str(message["content"]))
        for message in openai_style_messages
        if message["role"] in ["assistant", "user"] and message["content"]
    ]
