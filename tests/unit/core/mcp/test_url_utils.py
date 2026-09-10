"""Unit tests for MCP URL normalization helpers."""

import pytest

from src.app.core.mcp.url_utils import normalize_mcp_server_url, server_name_from_url


@pytest.mark.parametrize(
    ("hostname", "default_path", "expected"),
    [
        ("http://localhost:7001", "/mcp", "http://localhost:7001/mcp"),
        ("mcp:7001", "/mcp", "http://mcp:7001/mcp"),
        ("http://localhost:7001/sse", "/mcp", "http://localhost:7001/sse"),
        ("https://weather.example.com/mcp", "/mcp", "https://weather.example.com/mcp"),
    ],
)
def test_normalize_mcp_server_url(hostname: str, default_path: str, expected: str) -> None:
    assert normalize_mcp_server_url(hostname, default_path=default_path) == expected


def test_server_name_from_url_uses_host() -> None:
    assert server_name_from_url("http://localhost:7001/mcp", 0) == "localhost_7001"


def test_server_name_from_url_falls_back_to_index() -> None:
    assert server_name_from_url("", 3) == "server_3"
