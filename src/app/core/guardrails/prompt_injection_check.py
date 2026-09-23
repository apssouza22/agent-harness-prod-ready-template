"""Model-based prompt injection detection using ProtectAI DeBERTa classifier.

Complements the deterministic regex patterns in content_filter.py with
semantic classification via protectai/deberta-v3-base-prompt-injection-v2.

Requires the optional guardrails-ml dependencies:
    pip install -e ".[guardrails-ml]"
"""

import asyncio
from dataclasses import dataclass

from src.app.core.common.config import settings
from src.app.core.common.logging import logger

INJECTION_LABEL = "INJECTION"

_classifier = None


@dataclass
class PromptInjectionResult:
    """Result of model-based prompt injection classification."""

    is_injection: bool = False
    score: float = 0.0
    label: str = ""


def _get_classifier():
    """Lazy-initialize the DeBERTa prompt injection classifier."""
    global _classifier
    if _classifier is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        model_name = settings.GUARDRAIL_PROMPT_INJECTION_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=settings.GUARDRAIL_PROMPT_INJECTION_MAX_LENGTH,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    return _classifier


def _classify_sync(text: str, threshold: float) -> PromptInjectionResult:
    """Run synchronous classification against the DeBERTa model."""
    classifier = _get_classifier()
    predictions = classifier(text[: settings.GUARDRAIL_PROMPT_INJECTION_MAX_LENGTH])
    prediction = predictions[0]
    label = prediction["label"]
    score = float(prediction["score"])
    is_injection = label == INJECTION_LABEL and score >= threshold

    if is_injection:
        logger.warning(
            "prompt_injection_model_blocked",
            label=label,
            score=score,
            threshold=threshold,
        )

    return PromptInjectionResult(is_injection=is_injection, score=score, label=label)


async def detect_prompt_injection(
    text: str,
    *,
    threshold: float | None = None,
    enabled: bool | None = None,
) -> PromptInjectionResult:
    """Detect prompt injection attempts using the DeBERTa classifier.

    Args:
        text: The user input to classify.
        threshold: Confidence threshold for blocking. Defaults to settings value.
        enabled: Whether the model check is active. Defaults to settings value.

    Returns:
        PromptInjectionResult with classification outcome.
    """
    if not text or not text.strip():
        return PromptInjectionResult()

    is_enabled = (
        enabled if enabled is not None else settings.GUARDRAIL_PROMPT_INJECTION_MODEL_ENABLED
    )
    if not is_enabled:
        return PromptInjectionResult()

    effective_threshold = threshold if threshold is not None else settings.GUARDRAIL_PROMPT_INJECTION_THRESHOLD

    try:
        return await asyncio.to_thread(_classify_sync, text, effective_threshold)
    except Exception:
        logger.exception("prompt_injection_check_failed")
        return PromptInjectionResult()
