"""Open Deep Research agent module.

This package contains the Deep Research agent that conducts multi-step research
using a supervisor-researcher architecture with LangGraph subgraphs.
"""

from enum import Enum

from langchain.chat_models import init_chat_model

from src.app.agents.open_deep_research.state import ResearchQuestion, ClarifyWithUser
from src.app.agents.tools.search_tool import SearchAPI
from src.app.core.common.utils import get_api_key_for_model

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


MAX_CONTENT_LENGTH = 50000

# Shared configurable model used across all subgraphs
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


writer_model_config = {
    "model": FINAL_REPORT_MODEL,
    "max_tokens": FINAL_REPORT_MODEL_MAX_TOKENS,
    "api_key": get_api_key_for_model(FINAL_REPORT_MODEL),
}

research_model_config = {
    "model": RESEARCH_MODEL,
    "max_tokens": RESEARCH_MODEL_MAX_TOKENS,
    "api_key": get_api_key_for_model(RESEARCH_MODEL),
}
compress_model_config = {
    "model": COMPRESSION_MODEL,
    "max_tokens": COMPRESSION_MODEL_MAX_TOKENS,
    "api_key": get_api_key_for_model(COMPRESSION_MODEL),
}

synthesizer_model = init_chat_model().with_config(compress_model_config)
final_report_model = configurable_model.with_config(writer_model_config)

research_brief_model = (
    configurable_model
    .with_structured_output(ResearchQuestion)
    .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    .with_config(research_model_config)
)

clarification_model = (
    configurable_model
    .with_structured_output(ClarifyWithUser)
    .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    .with_config(research_model_config)
)


researcher_model = (
    configurable_model
    .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    .with_config(research_model_config)
)


supervisor_model = (
    configurable_model
    .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    .with_config(research_model_config)
)

