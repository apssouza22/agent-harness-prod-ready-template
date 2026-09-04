"""Input guardrail service — content filter and PII blocking."""

import time
from typing import Any, Optional

from src.app.core.guardrails.constants import (
    BLOCKED_INPUT_MESSAGE,
    BLOCKED_PII_MESSAGE,
    INPUT_BLOCK_PII_TYPES,
)
from src.app.core.guardrails.content_filter import check_content_filter
from src.app.core.guardrails.pii import PIIType, detect_pii
from src.app.core.guardrails.results import (
    GuardrailSource,
    InputBlockReason,
    InputGuardrailConfig,
    InputGuardrailResult,
)
from src.app.core.guardrails.tracing import guardrail_span
from src.app.core.common.logging import logger
from src.app.core.langfuse.client import LangfuseTracer
from src.app.core.metrics.metrics import (
    guardrail_check_duration_seconds,
    guardrail_checks_total,
    guardrail_pii_detections_total,
    guardrail_requests_blocked_total,
)


class InputGuardrail:
    """Validates user input with deterministic content filter and PII checks."""

    def __init__(
        self,
        config: InputGuardrailConfig | None = None,
        source: GuardrailSource = GuardrailSource.MIDDLEWARE,
        langfuse_tracer: Optional[LangfuseTracer] = None,
    ):
        self._config = config or InputGuardrailConfig()
        self._source = source
        self._langfuse_tracer = langfuse_tracer
        self._block_pii_types = self._config.block_pii_types or INPUT_BLOCK_PII_TYPES

    async def validate(
        self,
        content: str,
        *,
        trace: Any | None = None,
    ) -> InputGuardrailResult:
        """Run input guardrail checks on the given content."""
        if not content:
            return InputGuardrailResult()

        with guardrail_span(
            "guardrail_input_validation",
            tracer=self._langfuse_tracer,
            trace=trace,
            input_data={"content_length": len(content)},
            metadata={"guardrail_type": "input", "source": self._source.value},
        ) as span_result:
            if self._config.content_filter_enabled:
                filter_result = self._run_content_filter(content)
                if filter_result is not None:
                    span_result["output"] = {
                        "status": "blocked",
                        "check": InputBlockReason.CONTENT_FILTER.value,
                        "reason": filter_result.filter_reason,
                    }
                    return filter_result

            if self._config.pii_check_enabled:
                pii_result = self._run_pii_check(content)
                if pii_result is not None:
                    span_result["output"] = {
                        "status": "blocked",
                        "check": InputBlockReason.PII.value,
                        "pii_types": pii_result.pii_types,
                    }
                    return pii_result

            span_result["output"] = {"status": "passed"}

        return InputGuardrailResult()

    def _run_content_filter(self, content: str) -> InputGuardrailResult | None:
        start = time.perf_counter()
        filter_result = check_content_filter(
            content,
            banned_keywords=self._config.banned_keywords,
            check_prompt_injection=self._config.prompt_injection_check,
        )
        guardrail_check_duration_seconds.labels(
            guardrail_type="input", check_type="content_filter"
        ).observe(time.perf_counter() - start)

        if filter_result.is_blocked:
            guardrail_checks_total.labels(
                guardrail_type="input", check_type="content_filter", result="blocked"
            ).inc()
            guardrail_requests_blocked_total.labels(
                guardrail_type="input", reason="content_filter"
            ).inc()
            logger.info("input_guardrail_blocked", reason=filter_result.reason)
            return InputGuardrailResult(
                passed=False,
                block_reason=InputBlockReason.CONTENT_FILTER,
                blocked_message=BLOCKED_INPUT_MESSAGE,
                filter_reason=filter_result.reason,
            )

        guardrail_checks_total.labels(
            guardrail_type="input", check_type="content_filter", result="passed"
        ).inc()
        return None

    def _run_pii_check(self, content: str) -> InputGuardrailResult | None:
        start = time.perf_counter()
        pii_findings = detect_pii(content, pii_types=self._block_pii_types)
        guardrail_check_duration_seconds.labels(
            guardrail_type="input", check_type="pii"
        ).observe(time.perf_counter() - start)

        if not pii_findings:
            guardrail_checks_total.labels(
                guardrail_type="input", check_type="pii", result="passed"
            ).inc()
            return None

        detected_types = list({f["type"].value for f in pii_findings})
        for pii_type in detected_types:
            guardrail_pii_detections_total.labels(
                guardrail_type="input", pii_type=pii_type
            ).inc()
        guardrail_checks_total.labels(
            guardrail_type="input", check_type="pii", result="blocked"
        ).inc()
        guardrail_requests_blocked_total.labels(
            guardrail_type="input", reason="pii"
        ).inc()
        logger.info("input_guardrail_pii_blocked", pii_types=detected_types)
        return InputGuardrailResult(
            passed=False,
            block_reason=InputBlockReason.PII,
            blocked_message=BLOCKED_PII_MESSAGE,
            pii_types=detected_types,
        )
