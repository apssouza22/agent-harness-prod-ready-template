"""Guardrails module for agent input/output validation and safety.

Provides both deterministic and model-based guardrails following the
LangChain guardrails pattern:

- **Deterministic guardrails**: PII detection, content filtering, prompt injection detection
- **Model-based guardrails**: LLM safety evaluation for nuanced content checks

Usage with agent middleware:
    from src.app.core.middleware import GuardrailMiddleware

    pipeline = AgentPipeline(
        middlewares=[GuardrailMiddleware(), ...],
        invoke_fn=core_invoke,
    )

Standalone usage:
    from src.app.core.guardrails import check_content_filter, detect_pii, evaluate_safety

    filter_result = check_content_filter("user message")
    pii_findings = detect_pii("email: john@example.com")
    is_safe = await evaluate_safety("agent response")
"""

from src.app.core.guardrails.constants import BLOCKED_INPUT_MESSAGE, BLOCKED_PII_MESSAGE
from src.app.core.guardrails.content_filter import ContentFilterResult, check_content_filter
from src.app.core.guardrails.input_guardrail import InputGuardrail
from src.app.core.guardrails.output_guardrail import OutputGuardrail
from src.app.core.guardrails.pii import PIIStrategy, PIIType, apply_pii_strategy, detect_pii
from src.app.core.guardrails.results import (
    GuardrailSource,
    InputGuardrailConfig,
    InputGuardrailResult,
    OutputGuardrailConfig,
    OutputGuardrailResult,
)
from src.app.core.guardrails.safety_check import evaluate_safety, get_safe_replacement_message
