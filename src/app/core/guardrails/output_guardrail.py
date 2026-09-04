"""Output guardrail service — PII redaction and safety evaluation."""

import time
from typing import Any, Optional

from src.app.core.guardrails.constants import OUTPUT_REDACT_PII_TYPES
from src.app.core.guardrails.pii import PIIStrategy, apply_pii_strategy, detect_pii
from src.app.core.guardrails.results import (
    GuardrailSource,
    OutputAction,
    OutputGuardrailConfig,
    OutputGuardrailResult,
)
from src.app.core.guardrails.safety_check import evaluate_safety, get_safe_replacement_message
from src.app.core.guardrails.tracing import guardrail_span
from src.app.core.common.logging import logger
from src.app.core.langfuse.client import LangfuseTracer
from src.app.core.metrics.metrics import (
    guardrail_check_duration_seconds,
    guardrail_checks_total,
    guardrail_pii_detections_total,
    guardrail_requests_blocked_total,
)


class OutputGuardrail:
    """Validates agent output with PII redaction and model-based safety checks."""

    def __init__(
        self,
        config: OutputGuardrailConfig | None = None,
        source: GuardrailSource = GuardrailSource.MIDDLEWARE,
        langfuse_tracer: Optional[LangfuseTracer] = None,
    ):
        self._config = config or OutputGuardrailConfig()
        self._source = source
        self._langfuse_tracer = langfuse_tracer
        self._redact_pii_types = self._config.redact_pii_types or OUTPUT_REDACT_PII_TYPES
        self._pii_strategy = self._config.pii_strategy or PIIStrategy.REDACT

    async def validate(
        self,
        content: str,
        *,
        trace: Any | None = None,
    ) -> OutputGuardrailResult:
        """Run output guardrail checks on the given content."""
        if not content:
            return OutputGuardrailResult(content=content)

        modified_content = content
        actions: list[OutputAction] = []
        pii_types: list[str] = []

        with guardrail_span(
            "guardrail_output_validation",
            tracer=self._langfuse_tracer,
            trace=trace,
            input_data={"content_length": len(content)},
            metadata={"guardrail_type": "output", "source": self._source.value},
        ) as span_result:
            if self._config.pii_redact_enabled:
                modified_content, pii_action, detected = self._run_pii_redaction(modified_content)
                if pii_action:
                    actions.append(pii_action)
                    pii_types = detected

            if self._config.safety_check_enabled:
                modified_content, safety_action = await self._run_safety_check(modified_content)
                if safety_action:
                    actions.append(safety_action)

            span_result["output"] = {
                "status": "modified" if actions else "passed",
                "actions": [action.value for action in actions],
            }

        return OutputGuardrailResult(
            content=modified_content,
            modified=modified_content != content,
            actions=actions,
            pii_types=pii_types,
        )

    def _run_pii_redaction(
        self,
        content: str,
    ) -> tuple[str, OutputAction | None, list[str]]:
        start = time.perf_counter()
        pii_findings = detect_pii(content, pii_types=self._redact_pii_types)
        guardrail_check_duration_seconds.labels(
            guardrail_type="output", check_type="pii"
        ).observe(time.perf_counter() - start)

        if not pii_findings:
            guardrail_checks_total.labels(
                guardrail_type="output", check_type="pii", result="passed"
            ).inc()
            return content, None, []

        redacted = apply_pii_strategy(content, pii_findings, self._pii_strategy)
        if redacted is None:
            return content, None, []

        detected_types = list({f["type"].value for f in pii_findings})
        for pii_type in detected_types:
            guardrail_pii_detections_total.labels(
                guardrail_type="output", pii_type=pii_type
            ).inc()
        guardrail_checks_total.labels(
            guardrail_type="output", check_type="pii", result="redacted"
        ).inc()
        guardrail_requests_blocked_total.labels(
            guardrail_type="output", reason="pii_redacted"
        ).inc()
        logger.info("output_guardrail_pii_redacted", pii_types=detected_types)
        return redacted, OutputAction.PII_REDACTED, detected_types

    async def _run_safety_check(self, content: str) -> tuple[str, OutputAction | None]:
        start = time.perf_counter()
        is_safe = await evaluate_safety(content)
        guardrail_check_duration_seconds.labels(
            guardrail_type="output", check_type="safety"
        ).observe(time.perf_counter() - start)

        if not is_safe:
            guardrail_checks_total.labels(
                guardrail_type="output", check_type="safety", result="blocked"
            ).inc()
            guardrail_requests_blocked_total.labels(
                guardrail_type="output", reason="safety"
            ).inc()
            logger.warning("output_guardrail_safety_blocked")
            return get_safe_replacement_message(), OutputAction.SAFETY_BLOCKED

        guardrail_checks_total.labels(
            guardrail_type="output", check_type="safety", result="passed"
        ).inc()
        return content, None
