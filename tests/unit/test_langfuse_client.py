"""Unit tests for LangfuseTracer wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from src.app.core.langfuse.client import LangfuseTracer


@pytest.fixture
def test_settings(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    from src.app.core.common import config as config_module

    return config_module.Settings()


def test_langfuse_tracer_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from src.app.core.common import config as config_module

    settings = config_module.Settings()
    tracer = LangfuseTracer(settings)

    assert tracer.client is None
    assert tracer.get_callback_handler() is None
    assert tracer.get_trace_id() is None
    assert tracer.submit_feedback("trace-1", 1.0) is False


def test_langfuse_tracer_submit_feedback(test_settings):
    mock_client = MagicMock()
    with patch("src.app.core.langfuse.client.Langfuse", return_value=mock_client):
        tracer = LangfuseTracer(test_settings)

    assert tracer.submit_feedback("trace-1", 0.5, comment="good") is True
    mock_client.create_score.assert_called_once_with(
        trace_id="trace-1",
        name="user-feedback",
        value=0.5,
        comment="good",
    )
