"""Middleware that applies input/output guardrails at the agent level.

Useful for agents that do not embed guardrails as LangGraph nodes
(e.g. TextSQLDeepAgent).
"""

from typing import Optional

from src.app.core.middleware.types import AgentContext, AgentMiddleware, InvokeResult
from src.app.core.common.logging import logger
from src.app.core.common.model.message import Message
from src.app.core.guardrails.content_filter import check_content_filter
from src.app.core.guardrails.pii import PIIStrategy, PIIType, apply_pii_strategy, detect_pii
from src.app.core.guardrails.safety_check import evaluate_safety, get_safe_replacement_message
from src.app.core.guardrails.tracing import guardrail_span

BLOCKED_INPUT_MESSAGE = (
    "I cannot process this request. Please rephrase your message and try again."
)
BLOCKED_PII_MESSAGE = (
    "Your message contains sensitive information (e.g., API keys, credentials). "
    "Please remove it and try again."
)

INPUT_BLOCK_PII_TYPES = [PIIType.API_KEY, PIIType.SSN, PIIType.CREDIT_CARD]
OUTPUT_REDACT_PII_TYPES = [
    PIIType.EMAIL, PIIType.CREDIT_CARD, PIIType.SSN,
    PIIType.PHONE, PIIType.API_KEY, PIIType.IP,
]


class GuardrailMiddleware(AgentMiddleware):
    """Applies configurable input/output guardrails."""

    def __init__(
        self,
        input_filter: bool = True,
        input_pii_block: bool = True,
        output_pii_redact: bool = True,
        output_safety_check: bool = True,
        block_pii_types: Optional[list[PIIType]] = None,
        redact_pii_types: Optional[list[PIIType]] = None,
    ):
        self._input_filter = input_filter
        self._input_pii_block = input_pii_block
        self._output_pii_redact = output_pii_redact
        self._output_safety_check = output_safety_check
        self._block_pii_types = block_pii_types or INPUT_BLOCK_PII_TYPES
        self._redact_pii_types = redact_pii_types or OUTPUT_REDACT_PII_TYPES

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        last_content = ctx.messages[-1].content if ctx.messages else ""

        with guardrail_span(
            "guardrail_input_validation",
            input_data={"content_length": len(last_content)},
            metadata={"guardrail_type": "input", "source": "middleware"},
            ctx=ctx,
        ) as span_result:
            if self._input_filter and last_content:
                filter_result = check_content_filter(last_content)
                if filter_result.is_blocked:
                    span_result["output"] = {
                        "status": "blocked",
                        "check": "content_filter",
                        "reason": filter_result.reason,
                    }
                    logger.info("middleware_input_guardrail_blocked", reason=filter_result.reason, session_id=ctx.session_id)
                    return [Message(role="assistant", content=BLOCKED_INPUT_MESSAGE)]

            if self._input_pii_block and last_content:
                pii_findings = detect_pii(last_content, pii_types=self._block_pii_types)
                if pii_findings:
                    detected_types = list({f["type"].value for f in pii_findings})
                    span_result["output"] = {
                        "status": "blocked",
                        "check": "pii",
                        "pii_types": detected_types,
                    }
                    logger.info("middleware_input_guardrail_pii_blocked", pii_types=detected_types, session_id=ctx.session_id)
                    return [Message(role="assistant", content=BLOCKED_PII_MESSAGE)]

            span_result["output"] = {"status": "passed"}

        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        if not result:
            return result

        last_msg = result[-1]
        if last_msg.role != "assistant":
            return result

        modified_content = last_msg.content

        with guardrail_span(
            "guardrail_output_validation",
            input_data={"content_length": len(modified_content)},
            metadata={"guardrail_type": "output", "source": "middleware"},
            ctx=ctx,
        ) as span_result:
            output_actions: list[str] = []

            if self._output_pii_redact:
                pii_findings = detect_pii(modified_content, pii_types=self._redact_pii_types)
                if pii_findings:
                    redacted = apply_pii_strategy(modified_content, pii_findings, PIIStrategy.REDACT)
                    if redacted is not None:
                        detected_types = list({f["type"].value for f in pii_findings})
                        output_actions.append("pii_redacted")
                        logger.info("middleware_output_guardrail_pii_redacted", pii_types=detected_types, session_id=ctx.session_id)
                        modified_content = redacted

            if self._output_safety_check:
                is_safe = await evaluate_safety(modified_content)
                if not is_safe:
                    output_actions.append("safety_blocked")
                    logger.warning("middleware_output_guardrail_safety_blocked", session_id=ctx.session_id)
                    modified_content = get_safe_replacement_message()

            if modified_content != last_msg.content:
                result = list(result)
                result[-1] = Message(role="assistant", content=modified_content)

            span_result["output"] = {
                "status": "modified" if output_actions else "passed",
                "actions": output_actions,
            }

        return result
