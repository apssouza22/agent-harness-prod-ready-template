"""Prompt templates for dialogue state tracking."""

DEFAULT_DIALOGUE_STATE_UPDATE_PROMPT = """You are a dialogue state tracker for a conversational assistant.
Your job is to maintain a structured snapshot of the current conversation within a single session.

Update the dialogue state based on the latest turn while preserving still-relevant information from the previous state.
Remove or replace fields that are no longer accurate.

Track these fields:
- topic: the main subject of the conversation
- active_goals: what the user is trying to accomplish right now
- slots: key-value facts established in this session (for example destination, date, budget)
- pending_clarifications: open questions the assistant still needs answered
- entities_in_focus: people, products, places, or tools currently relevant
- conversation_phase: one of greeting, information_gathering, task_execution, clarification, closing, or other
- summary: one or two sentences describing where the conversation stands

Guidelines:
- Use only information from the conversation transcript.
- Prefer updating existing slots instead of duplicating them.
- Clear pending clarifications once they are answered.
- Return JSON only with exactly these keys:
  topic, active_goals, slots, pending_clarifications, entities_in_focus, conversation_phase, summary
- Use empty strings, empty lists, or empty objects when a field has no value.
"""


def build_dialogue_state_update_prompt(previous_state_json: str, conversation: str) -> str:
    """Build the user prompt for updating dialogue state."""
    return (
        "Previous dialogue state:\n"
        f"{previous_state_json}\n\n"
        "Conversation transcript:\n"
        f"{conversation}\n\n"
        "Return the updated dialogue state as JSON."
    )
