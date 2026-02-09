"""Middleware for adding user_id and session_id to logging context."""

from typing import Callable

from fastapi import Request
from jose import (
    JWTError,
    jwt,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.app.core.common.config import settings
from src.app.core.common.logging import (
    bind_context,
    clear_context,
)


class LoggingContextMiddleware(BaseHTTPMiddleware):
    """Middleware for adding user_id and session_id to logging context."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Extract user_id and session_id from authenticated requests and add to logging context.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response from the application
        """
        try:
            # Clear any existing context from previous requests
            clear_context()

            # Extract token from Authorization header
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

                try:
                    # Decode token to get session_id (stored in "sub" claim)
                    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
                    session_id = payload.get("sub")

                    if session_id:
                        # Bind session_id to logging context
                        bind_context(session_id=session_id)

                        # Try to get user_id from request state after authentication
                        # This will be set by the dependency injection if the endpoint uses authentication
                        # We'll check after the request is processed

                except JWTError:
                    # Token is invalid, but don't fail the request - let the auth dependency handle it
                    pass

            # Process the request
            response = await call_next(request)

            # After request processing, check if user info was added to request state
            if hasattr(request.state, "user_id"):
                bind_context(user_id=request.state.user_id)

            return response

        finally:
            # Always clear context after request is complete to avoid leaking to other requests
            clear_context()
