"""Unit tests for evaluation helpers."""

import json
from types import SimpleNamespace

from src.evals.helpers import get_input_output


def _observation(output):
    return SimpleNamespace(output=output)


def test_get_input_output_parses_json_string_messages():
    payload = {
        "messages": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi there"},
        ]
    }
    observation = _observation(json.dumps(payload))

    input_text, output_text = get_input_output(observation)

    assert input_text == "human: Hello"
    assert output_text == "ai: Hi there"


def test_get_input_output_returns_none_for_non_dict_output():
    observation = _observation("plain text response")

    input_text, output_text = get_input_output(observation)

    assert input_text is None
    assert output_text is None
