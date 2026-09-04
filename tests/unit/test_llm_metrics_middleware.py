import time
from unittest.mock import MagicMock

import pytest

from src.app.core.metrics.middleware import LlmMetricsMiddleware
from src.app.core.middleware.types import AgentContext


@pytest.mark.asyncio
async def test_llm_metrics_middleware_records_duration_and_tokens(monkeypatch):
    observed = []
    incremented = {"in": 0, "out": 0}

    class FakeHistogram:
        def labels(self, **kwargs):
            self._labels = kwargs
            return self

        def observe(self, value):
            observed.append((self._labels, value))

    class FakeCounter:
        def labels(self, **kwargs):
            self._labels = kwargs
            return self

        def inc(self, value):
            if "input" in str(self._labels):
                incremented["in"] += value
            else:
                incremented["out"] += value

    monkeypatch.setattr(
        "src.app.core.metrics.middleware.llm_inference_duration_seconds",
        FakeHistogram(),
    )
    monkeypatch.setattr(
        "src.app.core.metrics.middleware.record_token_usage",
        lambda response, model, agent_name: None,
    )

    middleware = LlmMetricsMiddleware()
    ctx = AgentContext(
        messages=[],
        session_id="session-1",
        user_id=1,
        config={},
        agent_name="chatbot",
    )

    await middleware.before_model_call(ctx, messages=[], model_name="gpt-4o")
    time.sleep(0.01)
    response = MagicMock()
    await middleware.after_model_call(ctx, response=response, model_name="gpt-4o")

    assert len(observed) == 1
    assert observed[0][0]["model"] == "gpt-4o"
    assert observed[0][0]["agent_name"] == "chatbot"
    assert observed[0][1] > 0
