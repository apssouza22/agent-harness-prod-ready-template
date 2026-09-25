"""Dialogue state service factory."""

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.dialogue_state.service import DialogueStateService


def make_dialogue_state_service(app_settings: Settings | None = None) -> DialogueStateService:
    """Create a dialogue state service instance."""
    resolved_settings = app_settings or default_settings
    return DialogueStateService(resolved_settings)


def make_dialogue_state_service_fresh(app_settings: Settings | None = None) -> DialogueStateService:
    """Create a new dialogue state service instance.

    Useful in tests that need an isolated service.
    """
    return make_dialogue_state_service(app_settings)
