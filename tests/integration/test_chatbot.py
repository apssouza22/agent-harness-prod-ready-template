"""Integration tests for chatbot endpoints.

All OpenAI / LangGraph calls are mocked via the patched agent factories
in conftest.py, so these tests exercise the HTTP layer, auth, rate-limit
wiring, and response serialisation without making real LLM requests.
"""

import json

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# POST /chatbot/chat
# ---------------------------------------------------------------------------

class TestChat:
    async def test_chat_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/chatbot/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1
        assert data["messages"][0]["role"] == "assistant"

    async def test_chat_no_auth(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/chatbot/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401

    async def test_chat_empty_messages(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/chatbot/chat",
            json={"messages": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_chat_missing_content(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/chatbot/chat",
            json={"messages": [{"role": "user"}]},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_chat_invalid_role(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/chatbot/chat",
            json={"messages": [{"role": "admin", "content": "Hello"}]},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_chat_xss_content_rejected(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/chatbot/chat",
            json={"messages": [{"role": "user", "content": "<script>alert('xss')</script>"}]},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /chatbot/chat/stream
# ---------------------------------------------------------------------------

class TestChatStream:
    async def test_stream_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/chatbot/chat/stream",
            json={"messages": [{"role": "user", "content": "Stream me a story"}]},
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
            "/api/v1/chatbot/chat/stream",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /chatbot/messages
# ---------------------------------------------------------------------------

class TestGetMessages:
    async def test_get_messages_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.get(
            "/api/v1/chatbot/messages",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)

    async def test_get_messages_no_auth(self, client: AsyncClient):
        response = await client.get("/api/v1/chatbot/messages")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /chatbot/messages
# ---------------------------------------------------------------------------

class TestClearMessages:
    async def test_clear_messages_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.delete(
            "/api/v1/chatbot/messages",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()

    async def test_clear_messages_no_auth(self, client: AsyncClient):
        response = await client.delete("/api/v1/chatbot/messages")
        assert response.status_code == 401
