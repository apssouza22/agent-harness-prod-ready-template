"""Utilities for MCP tool handling."""

from typing import Any, Callable, Optional

from anyio import ClosedResourceError
from langchain_core.messages import ToolMessage

from src.app.core.common.logging import bind_context, logger
from src.app.core.mcp.session_manager import generate_correlation_id, get_mcp_session_manager
from src.app.core.metrics.metrics import tool_executions_total


async def handle_mcp_tool_call(
    tool_fn: Callable,
    tool_call: dict[str, Any],
    tool_name: str,
    max_retries: int = 1,
    on_reconnect: Optional[Callable] = None,
    correlation_id: Optional[str] = None,
) -> ToolMessage:
    """Handle MCP tool calls with reconnection logic.

    Args:
        tool_fn: The tool function to invoke.
        tool_call: Dictionary containing tool call information with keys:
            - "args": Arguments to pass to the tool
            - "id": Tool call ID
        tool_name: Name of the tool being called.
        max_retries: Maximum number of reconnection attempts. Defaults to 1.
        on_reconnect: Optional callback to execute after successful reconnection
            (e.g., to reload tools).
        correlation_id: Optional correlation ID for tracking this tool call across logs.

    Returns:
        ToolMessage: A ToolMessage containing either the tool result or error information.
    """
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = generate_correlation_id()

    # Bind correlation ID to logging context
    bind_context(mcp_tool_correlation_id=correlation_id, mcp_tool_name=tool_name)

    logger.info(
        "mcp_tool_call_started",
        correlation_id=correlation_id,
        tool_name=tool_name,
        tool_call_id=tool_call["id"],
    )

    for attempt in range(max_retries + 1):
        try:
            tool_result = await tool_fn.ainvoke(tool_call["args"])
            tool_executions_total.labels(tool_name=tool_name, status="success").inc()
            logger.info(
                "mcp_tool_call_successful",
                correlation_id=correlation_id,
                tool_name=tool_name,
                tool_call_id=tool_call["id"],
                attempt=attempt + 1,
            )
            return ToolMessage(
                content=tool_result,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )

        except Exception as tool_error:
            # Check if it's a ClosedResourceError and we can retry
            if isinstance(tool_error, ClosedResourceError) and attempt < max_retries:
                logger.warning(
                    "mcp_connection_closed_retrying",
                    correlation_id=correlation_id,
                    tool_name=tool_name,
                    tool_call_id=tool_call["id"],
                    attempt=attempt + 1,
                    error=str(tool_error),
                )

                # Attempt to reconnect MCP sessions
                try:
                    mcp_manager = get_mcp_session_manager()
                    reconnected = await mcp_manager.reconnect()
                    if reconnected:
                        logger.info(
                            "mcp_reconnected_retrying_tool",
                            correlation_id=correlation_id,
                            tool_name=tool_name,
                        )
                        # Call the reconnection callback if provided
                        if on_reconnect:
                            await on_reconnect()
                        continue  # Retry the tool call
                except Exception as reconnect_error:
                    logger.error(
                        "mcp_reconnection_failed",
                        correlation_id=correlation_id,
                        error=str(reconnect_error),
                    )

            # Either not a ClosedResourceError, out of retries, or reconnection failed
            logger.error(
                "mcp_tool_call_failed",
                correlation_id=correlation_id,
                error=str(tool_error),
                error_type=type(tool_error).__name__,
                tool_name=tool_name,
                tool_call_id=tool_call["id"],
                attempt=attempt + 1,
            )

            tool_executions_total.labels(tool_name=tool_name, status="error").inc()
            error_msg = f"[ERROR] Tool '{tool_name}' failed: {str(tool_error)}"
            if isinstance(tool_error, ClosedResourceError):
                error_msg += " (MCP connection issue. Attempted reconnection.)"

            return ToolMessage(
                content=error_msg,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
