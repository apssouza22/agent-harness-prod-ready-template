"""Helpers for attaching LangChain callbacks to agent graph invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.app.core.middleware.types import AgentContext


def extend_graph_callbacks(ctx: AgentContext, callbacks: list[Any]) -> None:
    """Append callbacks to the graph invoke config without replacing existing ones."""
    graph_config = ctx.config.setdefault("graph_config", {})
    existing = graph_config.setdefault("callbacks", [])
    for callback in callbacks:
        if callback not in existing:
            existing.append(callback)
