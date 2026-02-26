"""Guardrails module for agent input/output validation and safety.

Provides both deterministic and model-based guardrails following the
LangChain guardrails pattern:

- **Deterministic guardrails**: PII detection, content filtering, prompt injection detection
- **Model-based guardrails**: LLM safety evaluation for nuanced content checks

Usage with LangGraph StateGraph:
    from src.app.agents.guardrails import create_input_guardrail_node, create_output_guardrail_node

    input_guardrail = create_input_guardrail_node(next_node="chat")
    output_guardrail = create_output_guardrail_node()

    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("output_guardrail", output_guardrail)

Standalone usage:
    from src.app.agents.guardrails import check_content_filter, detect_pii, evaluate_safety

    filter_result = check_content_filter("user message")
    pii_findings = detect_pii("email: john@example.com")
    is_safe = await evaluate_safety("agent response")
"""

from src.app.agents.guardrails.content_filter import ContentFilterResult, check_content_filter
from src.app.agents.guardrails.nodes import create_input_guardrail_node, create_output_guardrail_node
from src.app.agents.guardrails.pii import PIIStrategy, PIIType, apply_pii_strategy, detect_pii
from src.app.agents.guardrails.safety_check import evaluate_safety, get_safe_replacement_message
