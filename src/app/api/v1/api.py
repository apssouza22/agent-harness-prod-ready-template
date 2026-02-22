"""API v1 router configuration.

This module sets up the main API router and includes all sub-routers for different
endpoints like authentication and chatbot functionality.
"""

from datetime import datetime

from fastapi import APIRouter
from fastapi import (
    Request,
    status,
)
from starlette.responses import JSONResponse

from src.app.api.security.limiter import (
    limiter,
)
from src.app.api.v1.auth import router as auth_router
from src.app.api.v1.chatbot import router as chatbot_router
from src.app.api.v1.deep_research import router as deep_research_router
from src.app.api.v1.text_to_sql import router as text_to_sql_router
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.init import user_repository

api_router = APIRouter()

# Include routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
api_router.include_router(deep_research_router, prefix="/deep-research", tags=["deep-research"])
api_router.include_router(text_to_sql_router, prefix="/text-to-sql", tags=["text-to-sql"])


@api_router.get("/health")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint with environment-specific information.

    Returns:
        Dict[str, Any]: Health status information
    """
    logger.info("health_check_called")
    db_healthy = False
    try:
        user_repository.get_user(1)  # Simple query to check DB connectivity
        db_healthy = True
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response, status_code=status_code)


@api_router.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["root"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }
