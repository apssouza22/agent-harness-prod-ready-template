"""Dialogue state chat model construction via the shared LLM factory."""
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.dialogue_state.config_builder import resolve_dialogue_state_provider
from src.app.core.dialogue_state.models import DialogueStateLLMOutput
from src.app.core.llm.factory import make_chat_model, resolve_model_identifier


def make_dialogue_state_chat_model(app_settings: Settings | None = None) -> Runnable[PromptValue | str | Sequence[Any], dict[str, Any] | BaseModel]:
    """Create the structured-output chat model used for dialogue state updates."""
    resolved_settings = app_settings or default_settings
    provider = resolve_dialogue_state_provider(
        resolved_settings.DIALOGUE_STATE_LLM_PROVIDER,
        resolved_settings.DIALOGUE_STATE_MODEL,
        fallback_provider=resolved_settings.DEFAULT_LLM_PROVIDER,
    )
    model_name = resolve_model_identifier(
        resolved_settings.DIALOGUE_STATE_MODEL,
        provider,
        resolved_settings,
    )
    chat_model = make_chat_model(
        model_name,
        app_settings=resolved_settings,
        bifrost_agent="agent_1",
        max_tokens=resolved_settings.MAX_TOKENS,
    )
    return chat_model.with_structured_output(DialogueStateLLMOutput)
