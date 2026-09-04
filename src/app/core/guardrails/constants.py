"""Shared guardrail constants and default configuration values."""

from src.app.core.guardrails.pii import PIIType

BLOCKED_INPUT_MESSAGE = (
    "I cannot process this request. Please rephrase your message and try again."
)
BLOCKED_PII_MESSAGE = (
    "Your message contains sensitive information (e.g., API keys, credentials). "
    "Please remove it and try again."
)

INPUT_BLOCK_PII_TYPES: list[PIIType] = [
    PIIType.API_KEY,
    PIIType.SSN,
    PIIType.CREDIT_CARD,
]

OUTPUT_REDACT_PII_TYPES: list[PIIType] = [
    PIIType.EMAIL,
    PIIType.CREDIT_CARD,
    PIIType.SSN,
    PIIType.PHONE,
    PIIType.API_KEY,
    PIIType.IP,
]
