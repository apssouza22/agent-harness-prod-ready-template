"""Middleware that applies input/output guardrails at the agent level.

Useful for agents that do not embed guardrails as LangGraph nodes
(e.g. TextSQLDeepAgent).
"""

from typing import Optional

from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult
from src.app.core.common.logging import logger
from src.app.core.common.model.message import Message
from src.app.core.guardrails.constants import BLOCKED_INPUT_MESSAGE, BLOCKED_PII_MESSAGE
from src.app.core.guardrails.input_guardrail import InputGuardrail
from src.app.core.guardrails.output_guardrail import OutputGuardrail
from src.app.core.guardrails.pii import PIIType
from src.app.core.guardrails.results import (
    GuardrailSource,
    InputBlockReason,
    InputGuardrailConfig,
    OutputAction,
    OutputGuardrailConfig,
)
from src.app.core.langfuse.client import LangfuseTracer


class GuardrailMiddleware(AgentMiddleware):
    """Applies configurable input/output guardrails."""

    def __init__(
        self,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        input_filter: bool = True,
        input_pii_block: bool = True,
        output_pii_redact: bool = True,
        output_safety_check: bool = True,
        block_pii_types: Optional[list[PIIType]] = None,
        redact_pii_types: Optional[list[PIIType]] = None,
    ):
        self._input_guardrail = InputGuardrail(
            config=InputGuardrailConfig(
                content_filter_enabled=input_filter,
                pii_check_enabled=input_pii_block,
                block_pii_types=block_pii_types,
            ),
            source=GuardrailSource.MIDDLEWARE,
            langfuse_tracer=langfuse_tracer,
        )
        self._output_guardrail = OutputGuardrail(
            config=OutputGuardrailConfig(
                pii_redact_enabled=output_pii_redact,
                safety_check_enabled=output_safety_check,
                redact_pii_types=redact_pii_types,
            ),
            source=GuardrailSource.MIDDLEWARE,
            langfuse_tracer=langfuse_tracer,
        )

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        last_content = ctx.messages[-1].content if ctx.messages else ""
        trace = ctx.metadata.get("trace")
        result = await self._input_guardrail.validate(last_content, trace=trace)

        if not result.passed:
            if result.block_reason == InputBlockReason.CONTENT_FILTER:
                logger.info(
                    "middleware_input_guardrail_blocked",
                    reason=result.filter_reason,
                    session_id=ctx.session_id,
                )
                return [Message(role="assistant", content=BLOCKED_INPUT_MESSAGE)]
            if result.block_reason == InputBlockReason.PII:
                logger.info(
                    "middleware_input_guardrail_pii_blocked",
                    pii_types=result.pii_types,
                    session_id=ctx.session_id,
                )
                return [Message(role="assistant", content=BLOCKED_PII_MESSAGE)]

        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        if not result:
            return result

        last_msg = result[-1]
        if last_msg.role != "assistant":
            return result

        validation = await self._output_guardrail.validate(
            last_msg.content,
            trace=ctx.metadata.get("trace"),
        )

        if validation.modified:
            result = list(result)
            result[-1] = Message(role="assistant", content=validation.content)
            if OutputAction.PII_REDACTED in validation.actions:
                logger.info(
                    "middleware_output_guardrail_pii_redacted",
                    pii_types=validation.pii_types,
                    session_id=ctx.session_id,
                )
            if OutputAction.SAFETY_BLOCKED in validation.actions:
                logger.warning(
                    "middleware_output_guardrail_safety_blocked",
                    session_id=ctx.session_id,
                )

        return result
