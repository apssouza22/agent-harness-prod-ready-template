import asyncio
from enum import Enum
from typing import List

from langchain_core.tools import tool

from src.app.agents.tools import duckduckgo_search_tool
import logging


class SearchAPI(Enum):
    """Enumeration of available search API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DUCKDUCKGO = "duckduckgo"
    NONE = "none"


def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.

    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, DuckDuckGo, or None)

    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }]

    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]

    elif search_api == SearchAPI.DUCKDUCKGO:
        # Configure DuckDuckGo search tool with metadata
        search_tool = duckduckgo_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []

    # Default fallback for unknown search API types
    return []


##########################
# DuckDuckGo Search Tool Utils
##########################
DUCKDUCKGO_SEARCH_DESCRIPTION = (
    "A search engine for comprehensive web results. "
    "Useful for when you need to answer questions about current events."
)


@tool(description=DUCKDUCKGO_SEARCH_DESCRIPTION)
async def duckduckgo_search(queries: List[str]) -> str:
    """Execute multiple DuckDuckGo search queries and return formatted results.

    Args:
        queries: List of search queries to execute

    Returns:
        Formatted string containing search results from all queries
    """
    # Execute all search queries in parallel
    search_tasks = [duckduckgo_search_tool.ainvoke(query) for query in queries]

    try:
        search_results = await asyncio.gather(*search_tasks)
    except Exception:
        logging.warning("duckduckgo_search_failed", exc_info=True)
        return "Search failed. Please try different search queries."

    # Format the results
    if not any(search_results):
        return "No valid search results found. Please try different search queries."

    formatted_output = "Search results:\n\n"
    for i, (query, result) in enumerate(zip(queries, search_results)):
        formatted_output += f"--- QUERY {i + 1}: {query} ---\n"
        formatted_output += f"{result}\n"
        formatted_output += "-" * 80 + "\n\n"

    return formatted_output

