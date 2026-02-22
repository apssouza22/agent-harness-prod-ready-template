"""Utility functions and helpers for the Deep Research agent."""

from langchain_core.messages import (
    AIMessage,
    MessageLikeRepresentation,
)
from langchain_core.tools import tool

from src.app.agents.open_deep_research.config import (
    SEARCH_API,
)
from src.app.agents.open_deep_research.state import ResearchComplete
from src.app.agents.tools.search_tool import get_search_tool
from src.app.agents.tools.think_tool import think_tool


def get_all_tools():
    """Assemble complete toolkit including research and search tools.
    
    Returns:
        List of all configured and available tools for research operations
    """
    tools = [tool(ResearchComplete), think_tool]
    search_tools = get_search_tool(SEARCH_API)
    tools.extend(search_tools)
    return tools

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages
