"""Open Deep Research agent module.

This package contains the Deep Research agent that conducts multi-step research
using a supervisor-researcher architecture with LangGraph subgraphs.
"""

from enum import Enum

from langchain.chat_models import init_chat_model

from src.app.agents.tools.search_tool import SearchAPI

# ──────────────────────────────────────
# Deep Research Agent Configuration
# ──────────────────────────────────────

# General
ALLOW_CLARIFICATION = True
MAX_STRUCTURED_OUTPUT_RETRIES = 3

# Research limits
MAX_CONCURRENT_RESEARCH_UNITS = 5
MAX_RESEARCHER_ITERATIONS = 3
MAX_REACT_TOOL_CALLS = 10

# Search
SEARCH_API = SearchAPI.DUCKDUCKGO

# Models
RESEARCH_MODEL = "openai:gpt-4.1"
RESEARCH_MODEL_MAX_TOKENS = 10000

COMPRESSION_MODEL = "openai:gpt-4.1"
COMPRESSION_MODEL_MAX_TOKENS = 8192

FINAL_REPORT_MODEL = "openai:gpt-4.1"
FINAL_REPORT_MODEL_MAX_TOKENS = 10000

SUMMARIZATION_MODEL = "openai:gpt-4.1-mini"
SUMMARIZATION_MODEL_MAX_TOKENS = 8192

# Content
MAX_CONTENT_LENGTH = 50000

# Shared configurable model used across all subgraphs
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)
