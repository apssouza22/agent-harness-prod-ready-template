"""Unit tests for entity extraction parsing."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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


@pytest.mark.asyncio
async def test_extract_entities_forwards_bifrost_agent() -> None:
    extractor = EntityExtractor(settings)
    payload = {"entities": [{"name": "Acme Corp", "type": "organization"}]}

    mock_response = MagicMock()
    mock_response.content = json.dumps(payload)
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_response

    captured_kwargs: dict[str, object] = {}

    def capture_make_chat_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_llm

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "src.app.core.memory.entity_extractor.make_chat_model",
            capture_make_chat_model,
        )
        monkeypatch.setattr(settings, "LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED", True)
        result = await extractor.extract_entities("I work at Acme Corp")

    assert captured_kwargs.get("bifrost_agent") == "agent_1"
    assert len(result) == 1
    assert result[0].name == "Acme Corp"
