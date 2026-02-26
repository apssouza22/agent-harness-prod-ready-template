"""LangGraph guardrail node factories.

Provides factory functions to create input and output guardrail nodes
that can be inserted into any LangGraph StateGraph. Input guardrails
validate user messages before processing (deterministic). Output
guardrails validate agent responses before returning (deterministic + model-based).

Usage follows the LangChain guardrails middleware pattern adapted for
direct LangGraph StateGraph integration.
"""

from typing import Any, Callable

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command

from src.app.agents.guardrails.content_filter import ContentFilterResult, check_content_filter
from src.app.agents.guardrails.pii import PIIStrategy, PIIType, apply_pii_strategy, detect_pii
from src.app.agents.guardrails.safety_check import evaluate_safety, get_safe_replacement_message
from src.app.core.common.logging import logger

BLOCKED_INPUT_MESSAGE = (
    "I cannot process this request. Please rephrase your message and try again."
)
BLOCKED_PII_MESSAGE = (
    "Your message contains sensitive information (e.g., API keys, credentials). "
    "Please remove it and try again."
)


def _extract_messages(state: Any) -> list:
    """Extract messages from state regardless of Pydantic model or TypedDict."""
    if hasattr(state, "messages"):
        return state.messages
    if isinstance(state, dict):
        return state.get("messages", [])
    return []


def _get_last_user_content(messages: list) -> str:
    """Get the content of the last message (typically the user's latest input)."""
    if not messages:
        return ""
    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        return str(last_msg.content)
    if isinstance(last_msg, dict):
        return str(last_msg.get("content", ""))
    return str(last_msg)


def _get_last_ai_message(messages: list) -> tuple[Any | None, str]:
    """Get the last AI message object and its content string."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg, str(msg.content)
        if hasattr(msg, "type") and msg.type == "ai":
            return msg, str(msg.content)
    return None, ""


def create_input_guardrail_node(
    next_node: str,
    banned_keywords: list[str] | None = None,
    pii_check_enabled: bool = True,
    prompt_injection_check: bool = True,
    block_pii_types: list[PIIType] | None = None,
) -> Callable:
    """Create an input guardrail node for a LangGraph StateGraph.

    The returned async node validates user input before processing:
    1. Deterministic content filter (banned keywords + prompt injection)
    2. Deterministic PII detection (block strategy for sensitive types)

    If blocked, routes to END with a rejection message.
    Otherwise, routes to the specified next_node.

    Args:
        next_node: The graph node to route to when input passes validation.
        banned_keywords: Custom banned keywords. Defaults to built-in list.
        pii_check_enabled: Whether to check for PII in input.
        prompt_injection_check: Whether to check for prompt injection patterns.
        block_pii_types: PII types that trigger blocking. Defaults to API_KEY, SSN, CREDIT_CARD.

    Returns:
        An async node function compatible with LangGraph StateGraph.
    """
    if block_pii_types is None:
        block_pii_types = [PIIType.API_KEY, PIIType.SSN, PIIType.CREDIT_CARD]

    async def input_guardrail(state: Any) -> Command:
        messages = _extract_messages(state)
        content = _get_last_user_content(messages)

        if not content:
            return Command(goto=next_node)

        filter_result: ContentFilterResult = check_content_filter(
            content,
            banned_keywords=banned_keywords,
            check_prompt_injection=prompt_injection_check,
        )
        if filter_result.is_blocked:
            logger.info("input_guardrail_blocked", reason=filter_result.reason)
            return Command(
                update={"messages": [AIMessage(content=BLOCKED_INPUT_MESSAGE)]},
                goto=END,
            )

        if pii_check_enabled:
            pii_findings = detect_pii(content, pii_types=block_pii_types)
            if pii_findings:
                detected_types = list({f["type"].value for f in pii_findings})
                logger.info("input_guardrail_pii_blocked", pii_types=detected_types)
                return Command(
                    update={"messages": [AIMessage(content=BLOCKED_PII_MESSAGE)]},
                    goto=END,
                )

        return Command(goto=next_node)

    return input_guardrail


def create_output_guardrail_node(
    safety_check_enabled: bool = True,
    pii_redact_enabled: bool = True,
    redact_pii_types: list[PIIType] | None = None,
    pii_strategy: PIIStrategy = PIIStrategy.REDACT,
) -> Callable:
    """Create an output guardrail node for a LangGraph StateGraph.

    The returned async node validates agent output before returning to the user:
    1. Deterministic PII redaction on the response
    2. Model-based safety evaluation using an LLM

    If PII is found, it is redacted/masked per the strategy.
    If the response is flagged as unsafe, it is replaced with a safe message.

    Args:
        safety_check_enabled: Whether to run LLM-based safety evaluation.
        pii_redact_enabled: Whether to redact PII from output.
        redact_pii_types: PII types to redact. Defaults to common sensitive types.
        pii_strategy: Strategy for handling detected PII (redact, mask, hash).

    Returns:
        An async node function compatible with LangGraph StateGraph.
    """
    if redact_pii_types is None:
        redact_pii_types = [
            PIIType.EMAIL, PIIType.CREDIT_CARD, PIIType.SSN,
            PIIType.PHONE, PIIType.API_KEY, PIIType.IP,
        ]

    async def output_guardrail(state: Any) -> dict:
        messages = _extract_messages(state)
        last_ai_msg, content = _get_last_ai_message(messages)

        if not last_ai_msg or not content:
            return {"messages": []}

        modified_content = content

        if pii_redact_enabled:
            pii_findings = detect_pii(modified_content, pii_types=redact_pii_types)
            if pii_findings:
                redacted = apply_pii_strategy(modified_content, pii_findings, pii_strategy)
                if redacted is not None:
                    detected_types = list({f["type"].value for f in pii_findings})
                    logger.info("output_guardrail_pii_redacted", pii_types=detected_types)
                    modified_content = redacted

        if safety_check_enabled:
            is_safe = await evaluate_safety(modified_content)
            if not is_safe:
                logger.warning("output_guardrail_safety_blocked")
                modified_content = get_safe_replacement_message()

        if modified_content != content:
            replacement = AIMessage(content=modified_content, id=last_ai_msg.id)
            return {"messages": [replacement]}

        return {"messages": []}

    return output_guardrail
