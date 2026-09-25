"""Pydantic models for long-term memory structured LLM output."""

from pydantic import BaseModel, Field

from src.app.core.memory.entities import MemoryEntity, normalize_entity_name


class FactExtractionOutput(BaseModel):
    """Structured output for durable fact extraction."""

    facts: list[str] = Field(default_factory=list)


class EntityExtractionItem(BaseModel):
    """A named entity referenced in memory text."""

    name: str
    type: str = "unknown"


class EntityExtractionOutput(BaseModel):
    """Structured output for entity extraction."""

    entities: list[EntityExtractionItem] = Field(default_factory=list)

    def to_memory_entities(self) -> list[MemoryEntity]:
        """Convert structured entities into deduplicated memory entities."""
        results: list[MemoryEntity] = []
        seen: set[str] = set()

        for item in self.entities:
            name = item.name.strip()
            if not name:
                continue

            normalized = normalize_entity_name(name)
            if normalized in seen:
                continue

            seen.add(normalized)
            entity_type = item.type.strip() or "unknown"
            results.append(MemoryEntity(name=name, entity_type=entity_type))

        return results


class MemoryReconcileItem(BaseModel):
    """A single memory reconciliation action."""

    id: str = ""
    text: str = ""
    event: str = ""
    old_memory: str | None = None


class MemoryReconcileOutput(BaseModel):
    """Structured output for memory reconciliation."""

    memory: list[MemoryReconcileItem] = Field(default_factory=list)
