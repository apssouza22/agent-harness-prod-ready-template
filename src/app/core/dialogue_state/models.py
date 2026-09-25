"""Pydantic models for session-scoped dialogue state."""

from pydantic import BaseModel, Field


class DialogueSlot(BaseModel):
    """Key-value slot emitted by structured LLM output."""

    key: str
    value: str


class DialogueStateLLMOutput(BaseModel):
    """OpenAI-compatible structured output for dialogue state updates."""

    topic: str = ""
    active_goals: list[str] = Field(default_factory=list)
    slots: list[DialogueSlot] = Field(default_factory=list)
    pending_clarifications: list[str] = Field(default_factory=list)
    entities_in_focus: list[str] = Field(default_factory=list)
    conversation_phase: str = ""
    summary: str = ""

    def to_dialogue_state(self) -> "DialogueState":
        """Convert structured LLM output into persisted dialogue state."""
        return DialogueState(
            topic=self.topic,
            active_goals=self.active_goals,
            slots={slot.key: slot.value for slot in self.slots if slot.key.strip()},
            pending_clarifications=self.pending_clarifications,
            entities_in_focus=self.entities_in_focus,
            conversation_phase=self.conversation_phase,
            summary=self.summary,
        )


class DialogueState(BaseModel):
    """Structured conversation state for a single session."""

    topic: str = ""
    active_goals: list[str] = Field(default_factory=list)
    slots: dict[str, str] = Field(default_factory=dict)
    pending_clarifications: list[str] = Field(default_factory=list)
    entities_in_focus: list[str] = Field(default_factory=list)
    conversation_phase: str = ""
    summary: str = ""

    def is_empty(self) -> bool:
        """Return True when no tracked fields contain data."""
        return not any(
            [
                self.topic,
                self.active_goals,
                self.slots,
                self.pending_clarifications,
                self.entities_in_focus,
                self.conversation_phase,
                self.summary,
            ]
        )

    def to_prompt_text(self) -> str:
        """Format dialogue state for system prompt injection."""
        if self.is_empty():
            return "No structured dialogue state yet."

        lines: list[str] = []
        if self.topic:
            lines.append(f"- Topic: {self.topic}")
        if self.active_goals:
            lines.append(f"- Active goals: {', '.join(self.active_goals)}")
        if self.slots:
            slot_text = ", ".join(f"{key}={value}" for key, value in self.slots.items())
            lines.append(f"- Known slots: {slot_text}")
        if self.pending_clarifications:
            lines.append(f"- Pending clarifications: {', '.join(self.pending_clarifications)}")
        if self.entities_in_focus:
            lines.append(f"- Entities in focus: {', '.join(self.entities_in_focus)}")
        if self.conversation_phase:
            lines.append(f"- Conversation phase: {self.conversation_phase}")
        if self.summary:
            lines.append(f"- Summary: {self.summary}")
        return "\n".join(lines)
