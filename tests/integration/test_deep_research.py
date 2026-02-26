"""Integration tests for the deep research endpoints.

Agent factories are mocked in conftest.py so no real OpenAI calls are made.
"""

import json

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# POST /deep-research/research
# ---------------------------------------------------------------------------

class TestResearch:
    async def test_research_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/deep-research/research",
            json={"messages": [{"role": "user", "content": "Explain quantum computing"}]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1
        assert data["messages"][0]["role"] == "assistant"

    async def test_research_no_auth(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/deep-research/research",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401

    async def test_research_empty_messages(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/deep-research/research",
            json={"messages": []},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /deep-research/research/stream
# ---------------------------------------------------------------------------

class TestResearchStream:
    async def test_stream_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/deep-research/research/stream",
            json={"messages": [{"role": "user", "content": "Stream research on AI"}]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        chunks = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                chunks.append(payload)

        assert len(chunks) >= 2
        assert chunks[-1]["done"] is True

    async def test_stream_no_auth(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/deep-research/research/stream",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401
