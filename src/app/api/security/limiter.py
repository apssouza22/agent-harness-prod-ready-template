"""Rate limiting configuration for the application.

This module configures rate limiting using slowapi, with default limits
defined in the application settings. Rate limits are applied based on
remote IP addresses.
"""

from fastapi import FastAPI
from slowapi import (
    Limiter,
    _rate_limit_exceeded_handler,
)
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.app.core.common.config import settings

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=settings.RATE_LIMIT_DEFAULT)


def setup_rate_limit(app: FastAPI) -> None:
    """Set up rate limiter for the FastAPI application.

    Args:
        app: The FastAPI application instance
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
