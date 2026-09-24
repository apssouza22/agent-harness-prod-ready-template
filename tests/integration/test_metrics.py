"""Integration tests for Prometheus metrics endpoint."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestMetricsEndpoint:
    async def test_metrics_endpoint_returns_prometheus_payload(self, client: AsyncClient):
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "http_requests_total" in response.text

    async def test_api_request_does_not_crash_prometheus_middleware(self, client: AsyncClient):
        response = await client.get("/api/v1/")
        assert response.status_code == 200
