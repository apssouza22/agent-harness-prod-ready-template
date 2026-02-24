import asyncio
import os
from datetime import datetime

from langchain_core.messages import (
    MessageLikeRepresentation,
    filter_messages, ToolMessage,
)

from src.app.core.common.logging import logger
from src.app.core.metrics.metrics import tool_executions_total


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_api_key_for_model(model_name: str):
    """Get API key for a specific model from environment or config."""
    model_name = model_name.lower()
    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):
        return os.getenv("GOOGLE_API_KEY")
    return None


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    name = tool.name if hasattr(tool, "name") else tool.get("name", "unknown")
    try:
        logger.info("tool_call_started", tool_name=name, args=args)
        result = await tool.ainvoke(args, config)
        tool_executions_total.labels(tool_name=name, status="success").inc()
        return result
    except Exception as e:
        tool_executions_total.labels(tool_name=name, status="error").inc()
        return f"Error executing tool: {str(e)}"



async def execute_tools(config, most_recent_message, tools_by_name):
    """Execute tools called in the most recent message and return their outputs as ToolMessages."""
    tool_calls = most_recent_message.tool_calls
    if not tool_calls:
        return []
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]
    return tool_outputs