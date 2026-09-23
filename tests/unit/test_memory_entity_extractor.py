"""Unit tests for entity extraction parsing."""

from src.app.core.common.config import settings
from src.app.core.memory.entity_extractor import EntityExtractor


def test_parse_entities_deduplicates_and_normalizes() -> None:
    extractor = EntityExtractor(settings)
    payload = {
        "entities": [
            {"name": "Acme Corp", "type": "organization"},
            {"name": "acme corp", "type": "organization"},
            "Python",
            {"name": "", "type": "tool"},
        ]
    }

    entities = extractor._parse_entities(payload)

    assert len(entities) == 2
    assert entities[0].name == "Acme Corp"
    assert entities[1].name == "Python"
