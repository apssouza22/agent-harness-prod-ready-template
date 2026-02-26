"""Integration tests for authentication and session management endpoints."""

import pytest
from httpx import AsyncClient

from tests.integration.conftest import TEST_EMAIL, TEST_PASSWORD

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------

class TestRegister:
    async def test_register_success(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "new@example.com", "password": TEST_PASSWORD},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "new@example.com"
        assert data["id"] is not None
        assert data["token"]["access_token"]
        assert data["token"]["token_type"] == "bearer"

    async def test_register_duplicate_email(self, client: AsyncClient, registered_user):
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    async def test_register_weak_password(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "weak@example.com", "password": "short"},
        )
        assert response.status_code == 422

    async def test_register_invalid_email(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": TEST_PASSWORD},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

class TestLogin:
    async def test_login_success(self, client: AsyncClient, registered_user):
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": TEST_EMAIL, "password": TEST_PASSWORD, "grant_type": "password"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"]
        assert data["token_type"] == "bearer"

    async def test_login_wrong_password(self, client: AsyncClient, registered_user):
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": TEST_EMAIL, "password": "WrongPass123!", "grant_type": "password"},
        )
        assert response.status_code == 401

    async def test_login_nonexistent_user(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "nobody@example.com", "password": TEST_PASSWORD, "grant_type": "password"},
        )
        assert response.status_code == 401

    async def test_login_unsupported_grant_type(self, client: AsyncClient, registered_user):
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": TEST_EMAIL, "password": TEST_PASSWORD, "grant_type": "client_credentials"},
        )
        assert response.status_code == 400
        assert "grant type" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /auth/session
# ---------------------------------------------------------------------------

class TestCreateSession:
    async def test_create_session_success(self, client: AsyncClient, user_token):
        response = await client.post(
            "/api/v1/auth/session",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"]
        assert data["token"]["access_token"]

    async def test_create_session_no_auth(self, client: AsyncClient):
        response = await client.post("/api/v1/auth/session")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/sessions
# ---------------------------------------------------------------------------

class TestGetSessions:
    async def test_get_sessions(self, client: AsyncClient, user_token, session_with_token):
        response = await client.get(
            "/api/v1/auth/sessions",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 200
        sessions = response.json()
        assert isinstance(sessions, list)
        assert len(sessions) >= 1

    async def test_get_sessions_no_auth(self, client: AsyncClient):
        response = await client.get("/api/v1/auth/sessions")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# PATCH /auth/session/{session_id}/name
# ---------------------------------------------------------------------------

class TestUpdateSessionName:
    async def test_update_name_success(self, client: AsyncClient, session_with_token):
        sid = session_with_token["session_id"]
        token = session_with_token["token"]["access_token"]
        response = await client.patch(
            f"/api/v1/auth/session/{sid}/name",
            data={"name": "My Chat"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "My Chat"

    async def test_update_name_wrong_session(self, client: AsyncClient, session_with_token):
        token = session_with_token["token"]["access_token"]
        response = await client.patch(
            "/api/v1/auth/session/wrong-id/name",
            data={"name": "Hacked"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# DELETE /auth/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSession:
    async def test_delete_session_success(self, client: AsyncClient, session_with_token):
        sid = session_with_token["session_id"]
        token = session_with_token["token"]["access_token"]
        response = await client.delete(
            f"/api/v1/auth/session/{sid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    async def test_delete_session_wrong_id(self, client: AsyncClient, session_with_token):
        token = session_with_token["token"]["access_token"]
        response = await client.delete(
            "/api/v1/auth/session/wrong-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
