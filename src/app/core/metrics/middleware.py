"""Middleware that records LLM inference duration and token usage."""

import time
from typing import Any

from src.app.core.common.logging import logger
from src.app.core.metrics.metrics import llm_inference_duration_seconds
from src.app.core.metrics.token_usage import record_token_usage
from src.app.core.middleware.types import AgentContext, AgentMiddleware


class LlmMetricsMiddleware(AgentMiddleware):
    """Records Prometheus metrics for each LLM call in the agent graph."""

    _TIMER_STACK_KEY = "_llm_metrics_timer_stack"

    async def before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        timer_stack = ctx.metadata.setdefault(self._TIMER_STACK_KEY, [])
        timer_stack.append((model_name, time.perf_counter()))
        return messages

    async def after_model_call(
        self,
        ctx: AgentContext,
        *,
        response: Any,
        model_name: str,
    ) -> Any:
        timer_stack = ctx.metadata.get(self._TIMER_STACK_KEY, [])
        if not timer_stack:
            logger.debug("llm_metrics_timer_missing", model=model_name, agent_name=ctx.agent_name)
            record_token_usage(response, model_name, ctx.agent_name)
            return response

        started_model_name, started_at = timer_stack.pop()
        duration_seconds = time.perf_counter() - started_at
        llm_inference_duration_seconds.labels(
            model=started_model_name,
            agent_name=ctx.agent_name,
        ).observe(duration_seconds)
        record_token_usage(response, started_model_name, ctx.agent_name)
        return response
