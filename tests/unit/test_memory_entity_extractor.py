"""Unit tests for entity extraction parsing."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.common.config import settings
from src.app.core.memory.entity_extractor import EntityExtractor


def test_parse_entities_deduplicates_and_normalizes() -> None:
    extractor = EntityExtractor(settings, chat_model=MagicMock())
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
async def test_extract_entities_uses_injected_chat_model() -> None:
    payload = {"entities": [{"name": "Acme Corp", "type": "organization"}]}

    mock_response = MagicMock()
    mock_response.content = json.dumps(payload)
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_response
    extractor = EntityExtractor(settings, chat_model=mock_llm)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings, "LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED", True)
        result = await extractor.extract_entities("I work at Acme Corp")

    mock_llm.ainvoke.assert_awaited_once()
    assert len(result) == 1
    assert result[0].name == "Acme Corp"
