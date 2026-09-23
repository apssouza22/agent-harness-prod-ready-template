"""Dialogue state service factory."""

from functools import lru_cache

from src.app.core.common.config import Settings, settings as default_settings
from src.app.core.dialogue_state.service import DialogueStateService


def make_dialogue_state_service(app_settings: Settings | None = None) -> DialogueStateService:
    """Create a dialogue state service instance."""
    resolved_settings = app_settings or default_settings
    return DialogueStateService(resolved_settings)


@lru_cache(maxsize=1)
def make_dialogue_state_service_cached() -> DialogueStateService:
    """Return a process-wide cached dialogue state service using default settings."""
    return make_dialogue_state_service()


def make_dialogue_state_service_fresh(app_settings: Settings | None = None) -> DialogueStateService:
    """Create a new dialogue state service bypassing the process-wide cache."""
    return make_dialogue_state_service(app_settings)
