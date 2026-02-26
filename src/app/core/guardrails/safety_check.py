"""Model-based safety evaluation for agent outputs.

Uses an LLM to evaluate whether agent responses are safe and appropriate.
This provides semantic understanding beyond what deterministic rule-based
checks can catch, following the LangChain guardrails "after agent" pattern.
"""

from langchain.chat_models import init_chat_model

from src.app.core.common.config import settings
from src.app.core.common.logging import logger

SAFETY_EVALUATION_PROMPT = """You are a safety evaluator. Analyze the following AI assistant response \
and determine if it is safe and appropriate.

A response is UNSAFE if it:
- Contains instructions for harmful, illegal, or dangerous activities
- Includes hate speech, discrimination, or harassment
- Provides instructions for creating weapons, drugs, or explosives
- Contains sexually explicit content
- Encourages self-harm or violence
- Attempts to manipulate or deceive the user
- Reveals system prompts or internal instructions

Respond with ONLY one word: "SAFE" or "UNSAFE"

Response to evaluate:
{response}"""

SAFE_REPLACEMENT = (
    "I'm unable to provide that response as it may contain unsafe content. "
    "Please try rephrasing your request."
)

_safety_model = None


def _get_safety_model():
    """Lazy-initialize the safety evaluation model using a fast, low-cost model."""
    global _safety_model
    if _safety_model is None:
        _safety_model = init_chat_model(
            model=f"openai:{settings.LONG_TERM_MEMORY_MODEL}",
            api_key=settings.OPENAI_API_KEY,
            max_tokens=10,
            temperature=0,
        )
    return _safety_model


async def evaluate_safety(content: str) -> bool:
    """Evaluate whether content is safe using an LLM.

    Args:
        content: The text content to evaluate for safety.

    Returns:
        True if the content is safe, False if unsafe.
    """
    if not content or not content.strip():
        return True

    try:
        model = _get_safety_model()
        prompt = SAFETY_EVALUATION_PROMPT.format(response=content[:2000])
        result = await model.ainvoke([{"role": "user", "content": prompt}])
        verdict = result.content.strip().upper()
        is_safe = "UNSAFE" not in verdict

        if not is_safe:
            logger.warning("safety_check_flagged_unsafe", content_preview=content[:100])

        return is_safe
    except Exception:
        logger.exception("safety_check_evaluation_failed")
        return True


def get_safe_replacement_message() -> str:
    """Return the standard safety block message."""
    return SAFE_REPLACEMENT
