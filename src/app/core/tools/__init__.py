"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Currently includes tools for web search
and other external integrations.
"""

from langchain_core.tools.base import BaseTool

from .duckduckgo_search import duckduckgo_search_tool
from .search_tool import SearchAPI, get_search_tool

tools: list[BaseTool] = [duckduckgo_search_tool]

__ALL__ = ["tools", "SearchAPI", "get_search_tool"]
