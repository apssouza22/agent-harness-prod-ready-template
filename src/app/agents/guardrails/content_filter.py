"""Deterministic content filtering for input validation.

Provides keyword-based content filtering and prompt injection detection
to block inappropriate or malicious requests before they reach the agent.
Follows the LangChain guardrails "before agent" pattern for deterministic checks.
"""

import re
from dataclasses import dataclass, field

from src.app.core.common.logging import logger

DEFAULT_BANNED_KEYWORDS: list[str] = [
    "hack",
    "exploit",
    "malware",
    "ransomware",
    "phishing",
    "ddos",
    "keylogger",
    "rootkit",
    "trojan",
    "botnet",
]

PROMPT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(?:a|an|in)\s+(?:jailbreak|unrestricted|evil|DAN)",
    r"pretend\s+(?:you\s+(?:are|have)|to\s+be)\s+(?:a|an)\s+(?:jailbreak|unrestricted)",
    r"override\s+(?:your|all|system)\s+(?:instructions|rules|constraints)",
    r"system\s*prompt\s*(?:is|:)",
    r"reveal\s+(?:your|the|system)\s+(?:prompt|instructions|rules)",
]


@dataclass
class ContentFilterResult:
    """Result of content filtering check."""

    is_blocked: bool = False
    reason: str = ""
    matched_pattern: str = ""


def check_content_filter(
    text: str,
    banned_keywords: list[str] | None = None,
    check_prompt_injection: bool = True,
) -> ContentFilterResult:
    """Check text against content filter rules.

    Runs two layers of deterministic checks:
    1. Banned keyword matching
    2. Prompt injection pattern detection

    Args:
        text: The text to validate.
        banned_keywords: Custom banned keywords list. Defaults to DEFAULT_BANNED_KEYWORDS.
        check_prompt_injection: Whether to check for prompt injection patterns.

    Returns:
        ContentFilterResult: Result indicating if the text was blocked and why.
    """
    if not text:
        return ContentFilterResult()

    content_lower = text.lower()
    keywords = banned_keywords if banned_keywords is not None else DEFAULT_BANNED_KEYWORDS

    for keyword in keywords:
        if keyword.lower() in content_lower:
            logger.warning("content_filter_keyword_blocked", keyword=keyword)
            return ContentFilterResult(
                is_blocked=True,
                reason="Request contains prohibited content.",
                matched_pattern=keyword,
            )

    if check_prompt_injection:
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, content_lower):
                logger.warning("content_filter_prompt_injection_blocked", pattern=pattern)
                return ContentFilterResult(
                    is_blocked=True,
                    reason="Request contains a potential prompt injection attempt.",
                    matched_pattern=pattern,
                )

    return ContentFilterResult()
