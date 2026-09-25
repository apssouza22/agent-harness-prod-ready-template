"""Dialogue state service factory."""

from langchain_core.language_models.chat_models import BaseChatModel

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.dialogue_state.llm import make_dialogue_state_chat_model
from src.app.core.dialogue_state.service import DialogueStateService


def make_dialogue_state_service(
    app_settings: Settings | None = None,
    chat_model: BaseChatModel | None = None,
) -> DialogueStateService:
    """Create a dialogue state service instance."""
    resolved_settings = app_settings or default_settings
    llm = chat_model or make_dialogue_state_chat_model(resolved_settings)
    return DialogueStateService(resolved_settings, chat_model=llm)


def make_dialogue_state_service_fresh(
    app_settings: Settings | None = None,
    chat_model: BaseChatModel | None = None,
) -> DialogueStateService:
    """Create a new dialogue state service instance.

    Useful in tests that need an isolated service.
    """
    return make_dialogue_state_service(app_settings, chat_model=chat_model)
