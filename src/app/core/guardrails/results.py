"""Result types returned by guardrail service classes."""

from dataclasses import dataclass, field
from enum import Enum

from src.app.core.guardrails.pii import PIIStrategy, PIIType


class GuardrailSource(str, Enum):
    """Where the guardrail check is executed."""

    GRAPH_NODE = "graph_node"
    MIDDLEWARE = "middleware"


class InputBlockReason(str, Enum):
    """Reason an input guardrail blocked a request."""

    CONTENT_FILTER = "content_filter"
    PII = "pii"


class OutputAction(str, Enum):
    """Action taken by the output guardrail."""

    PII_REDACTED = "pii_redacted"
    SAFETY_BLOCKED = "safety_blocked"


@dataclass(frozen=True)
class InputGuardrailConfig:
    """Configuration for input validation."""

    content_filter_enabled: bool = True
    banned_keywords: list[str] | None = None
    pii_check_enabled: bool = True
    prompt_injection_check: bool = True
    block_pii_types: list[PIIType] | None = None


@dataclass(frozen=True)
class OutputGuardrailConfig:
    """Configuration for output validation."""

    safety_check_enabled: bool = True
    pii_redact_enabled: bool = True
    redact_pii_types: list[PIIType] | None = None
    pii_strategy: PIIStrategy = PIIStrategy.REDACT


@dataclass
class InputGuardrailResult:
    """Outcome of input guardrail validation."""

    passed: bool = True
    block_reason: InputBlockReason | None = None
    blocked_message: str | None = None
    filter_reason: str | None = None
    pii_types: list[str] = field(default_factory=list)


@dataclass
class OutputGuardrailResult:
    """Outcome of output guardrail validation."""

    content: str
    modified: bool = False
    actions: list[OutputAction] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)
