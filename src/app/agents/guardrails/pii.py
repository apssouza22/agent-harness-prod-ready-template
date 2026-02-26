"""PII detection and handling utilities.

Provides deterministic PII detection using regex patterns for common PII types
including emails, credit cards, IP addresses, and more. Supports multiple
handling strategies following the LangChain guardrails pattern:
redact, mask, hash, and block.
"""

import hashlib
import re
from enum import Enum

from src.app.core.common.logging import logger


class PIIType(str, Enum):
    """Supported PII types for detection."""

    EMAIL = "email"
    CREDIT_CARD = "credit_card"
    IP = "ip"
    URL = "url"
    MAC_ADDRESS = "mac_address"
    API_KEY = "api_key"
    PHONE = "phone"
    SSN = "ssn"


class PIIStrategy(str, Enum):
    """Strategies for handling detected PII."""

    REDACT = "redact"
    MASK = "mask"
    HASH = "hash"
    BLOCK = "block"


PII_PATTERNS: dict[PIIType, str] = {
    PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    PIIType.CREDIT_CARD: r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    PIIType.IP: r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b",
    PIIType.URL: r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w.?&=%-]*",
    PIIType.MAC_ADDRESS: r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
    PIIType.API_KEY: r"\b(?:sk|pk|api[_-]?key)[_-]?[a-zA-Z0-9]{20,}\b",
    PIIType.PHONE: r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    PIIType.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
}


def detect_pii(text: str, pii_types: list[PIIType] | None = None) -> list[dict]:
    """Detect PII in text using regex patterns.

    Args:
        text: The text to scan for PII.
        pii_types: Specific PII types to check. Defaults to all types.

    Returns:
        list[dict]: List of findings with type, value, start, and end positions.
    """
    if not text:
        return []

    types_to_check = pii_types if pii_types is not None else list(PIIType)
    findings = []

    for pii_type in types_to_check:
        pattern = PII_PATTERNS.get(pii_type)
        if not pattern:
            continue
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if pii_type == PIIType.CREDIT_CARD and not _luhn_check(match.group()):
                continue
            findings.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

    if findings:
        detected_types = list({f["type"].value for f in findings})
        logger.info("pii_detected", pii_types=detected_types, count=len(findings))

    return findings


def apply_pii_strategy(text: str, findings: list[dict], strategy: PIIStrategy) -> str | None:
    """Apply a PII handling strategy to the text.

    Args:
        text: The original text containing PII.
        findings: PII findings from detect_pii().
        strategy: The strategy to apply (redact, mask, hash, block).

    Returns:
        The processed text, or None if strategy is BLOCK.
    """
    if not findings:
        return text

    if strategy == PIIStrategy.BLOCK:
        return None

    sorted_findings = sorted(findings, key=lambda f: f["start"], reverse=True)
    result = text

    for finding in sorted_findings:
        original = finding["value"]
        pii_type = finding["type"]

        if strategy == PIIStrategy.REDACT:
            replacement = f"[REDACTED_{pii_type.value.upper()}]"
        elif strategy == PIIStrategy.MASK:
            replacement = _mask_value(original, pii_type)
        elif strategy == PIIStrategy.HASH:
            replacement = hashlib.sha256(original.encode()).hexdigest()[:12]
        else:
            replacement = original

        result = result[:finding["start"]] + replacement + result[finding["end"]:]

    return result


def _luhn_check(card_number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = re.sub(r"[\s-]", "", card_number)
    if not digits.isdigit() or len(digits) < 13:
        return False

    total = 0
    for i, digit in enumerate(reversed(digits)):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _mask_value(value: str, pii_type: PIIType) -> str:
    """Partially mask a PII value based on its type."""
    if pii_type == PIIType.EMAIL:
        parts = value.split("@")
        if len(parts) == 2:
            return f"{parts[0][:2]}****@{parts[1]}"
    elif pii_type == PIIType.CREDIT_CARD:
        digits = re.sub(r"[\s-]", "", value)
        return f"****-****-****-{digits[-4:]}"
    elif pii_type == PIIType.PHONE:
        digits = re.sub(r"\D", "", value)
        return f"****-****-{digits[-4:]}"
    elif pii_type == PIIType.SSN:
        return f"***-**-{value[-4:]}"

    if len(value) > 4:
        return "*" * (len(value) - 4) + value[-4:]
    return "*" * len(value)
