"""Unit tests for entity extraction parsing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.app.core.common.config import settings
from src.app.core.memory.entity_extractor import EntityExtractor
from src.app.core.memory.models import EntityExtractionItem, EntityExtractionOutput


def test_entity_extraction_output_deduplicates_and_normalizes() -> None:
    output = EntityExtractionOutput(
        entities=[
            EntityExtractionItem(name="Acme Corp", type="organization"),
            EntityExtractionItem(name="acme corp", type="organization"),
            EntityExtractionItem(name="Python", type="tool"),
            EntityExtractionItem(name="", type="tool"),
        ]
    )

    entities = output.to_memory_entities()

    assert len(entities) == 2
    assert entities[0].name == "Acme Corp"
    assert entities[1].name == "Python"


@pytest.mark.asyncio
async def test_extract_entities_uses_injected_chat_model() -> None:
    mock_wrapped = AsyncMock()
    mock_wrapped.ainvoke.return_value = EntityExtractionOutput(
        entities=[EntityExtractionItem(name="Acme Corp", type="organization")]
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_wrapped
    extractor = EntityExtractor(settings, chat_model=mock_llm)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings, "LONG_TERM_MEMORY_ENTITY_BOOST_ENABLED", True)
        result = await extractor.extract_entities("I work at Acme Corp")

    mock_llm.with_structured_output.assert_called_once()
    mock_wrapped.ainvoke.assert_awaited_once()
    assert len(result) == 1
    assert result[0].name == "Acme Corp"
