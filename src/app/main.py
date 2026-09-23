"""This file contains the main application entry point."""

from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.app.agents.chatbot.factory import make_chatbot_agent
from src.app.agents.open_deep_research.factory import make_deep_research_agent
from src.app.agents.text_to_sql.factory import make_text_to_sql_agent
from src.app.api.logging_context import LoggingContextMiddleware
from src.app.api.metrics.http_metrics import setup_metrics
from src.app.api.metrics.middleware import MetricsMiddleware
from src.app.api.security.limiter import setup_rate_limit
from src.app.api.v1.api import api_router
from src.app.core.cache.embeddings import OpenAIEmbeddingsClient
from src.app.core.cache.factory import make_cache_client
from src.app.core.checkpoint.factory import make_checkpoint_service
from src.app.core.db.connection_pool import get_connection_pool, reset_connection_pool
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.db.factory import make_database
from src.app.core.dialogue_state.factory import make_dialogue_state_service
from src.app.core.memory.factory import make_memory_service
from src.app.core.langfuse.factory import make_langfuse_tracer
from src.app.core.session.factory import make_session_repository
from src.app.core.user.factory import make_user_repository
from src.app.core.mcp.dependencies import cleanup_mcp_manager, initialize_mcp_manager
from src.app.core.mcp.factory import make_mcp_manager

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wire the full dependency graph onto app.state at startup."""
    app.state.settings = settings

    langfuse_tracer = make_langfuse_tracer()
    app.state.langfuse_tracer = langfuse_tracer

    database = make_database(settings)
    app.state.database = database

    db_session = database.get_session_maker()
    app.state.user_repository = make_user_repository(db_session)
    app.state.session_repository = make_session_repository(db_session)

    app.state.memory_service = make_memory_service(settings)
    app.state.dialogue_state_service = make_dialogue_state_service(settings)

    embeddings_client = OpenAIEmbeddingsClient(settings) if settings.CACHE_SEMANTIC_ENABLED else None
    app.state.cache_client = make_cache_client(settings, embeddings_client=embeddings_client)

    connection_pool = await get_connection_pool(settings)
    checkpoint_service = make_checkpoint_service(settings, connection_pool=connection_pool)
    app.state.checkpoint_service = checkpoint_service

    checkpointer = await checkpoint_service.get_checkpointer()
    app.state.checkpointer = checkpointer

    mcp_manager = make_mcp_manager()
    app.state.mcp_manager = mcp_manager
    await initialize_mcp_manager(mcp_manager)

    app.state.chatbot_agent = await make_chatbot_agent(
        checkpointer,
        langfuse_tracer=langfuse_tracer,
        mcp_manager=mcp_manager,
    )
    app.state.deep_research_agent = await make_deep_research_agent(checkpointer, langfuse_tracer=langfuse_tracer)
    app.state.text_to_sql_agent = await make_text_to_sql_agent(langfuse_tracer=langfuse_tracer)

    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        api_prefix=settings.API_V1_STR,
    )

    yield

    await cleanup_mcp_manager(mcp_manager)
    await reset_connection_pool()
    langfuse_tracer.shutdown()
    database.dispose()

    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set up Prometheus metrics
setup_metrics(app)
setup_rate_limit(app)

# Add logging context middleware (must be added before other middleware to capture context)
app.add_middleware(LoggingContextMiddleware)
app.add_middleware(MetricsMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data."""
    logger.error(
        "validation_error",
        client_host=request.client.host if request.client else "unknown",
        path=request.url.path,
        errors=str(exc.errors()),
    )

    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join([str(loc_part) for loc_part in error["loc"] if loc_part != "body"])
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
