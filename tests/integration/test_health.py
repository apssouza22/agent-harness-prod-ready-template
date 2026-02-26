"""Integration tests for health check and root endpoints."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestHealthCheck:
    async def test_health_endpoint(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data
        assert data["components"]["api"] == "healthy"


class TestRoot:
    async def test_root_endpoint(self, client: AsyncClient):
        response = await client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "name" in data
        assert data["swagger_url"] == "/docs"
