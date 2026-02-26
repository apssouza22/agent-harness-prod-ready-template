"""Integration tests for the text-to-SQL endpoint.

The TextSQLDeepAgent is mocked in conftest.py so no real OpenAI or
database calls are made.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# POST /text-to-sql/query
# ---------------------------------------------------------------------------

class TestTextToSQLQuery:
    async def test_query_success(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/text-to-sql/query",
            json={"query": "How many users signed up last week?"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1

    async def test_query_no_auth(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/text-to-sql/query",
            json={"query": "SELECT 1"},
        )
        assert response.status_code == 401

    async def test_query_empty_string(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/text-to-sql/query",
            json={"query": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_query_missing_field(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/text-to-sql/query",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_query_too_long(self, client: AsyncClient, auth_headers: dict):
        response = await client.post(
            "/api/v1/text-to-sql/query",
            json={"query": "x" * 2001},
            headers=auth_headers,
        )
        assert response.status_code == 422
