"""Deep Research API endpoints for submitting research queries.

This module provides endpoints for deep research interactions, including
submitting a research query and streaming the research report.
"""

import json

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

from src.app.agents.open_deep_research import get_deep_research_agent
from src.app.api.metrics.http_metrics import llm_stream_duration_seconds
from src.app.api.security.limiter import limiter
from src.app.api.v1.auth import get_current_session
from src.app.api.v1.dtos.chat import (
    ChatRequest,
    ChatResponse,
    StreamResponse,
)
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.session.session_model import Session

router = APIRouter()


@router.post("/research", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["deep_research"][0])
async def research(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Submit a deep research query.

    Accepts a user message describing the research topic and returns a
    comprehensive research report generated through multi-step research.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The request containing the research query messages.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The research report as a list of messages.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        logger.info(
            "deep_research_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        agent = await get_deep_research_agent()
        result = await agent.agent_invoke(chat_request.messages, session.id, user_id=session.user_id)

        logger.info("deep_research_request_processed", session_id=session.id)

        return ChatResponse(messages=result)
    except Exception as e:
        logger.error("deep_research_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["deep_research_stream"][0])
async def research_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Submit a deep research query with streaming response.

    Streams the research report as server-sent events while the multi-step
    research workflow runs.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The request containing the research query messages.
        session: The current session from the auth token.

    Returns:
        StreamingResponse: A streaming response of the research report.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        agent = await get_deep_research_agent()
        logger.info(
            "deep_research_stream_request_received",
            session_id=session.id,
            message_count=len(chat_request.messages),
        )

        async def event_generator():
            """Generate streaming events.

            Yields:
                str: Server-sent events in JSON format.
            """
            try:
                with llm_stream_duration_seconds.labels(model="deep_research").time():
                    async for chunk in agent.agent_invoke_stream(
                        chat_request.messages, session.id, user_id=session.user_id
                    ):
                        response = StreamResponse(content=chunk, done=False)
                        yield f"data: {json.dumps(response.model_dump())}\n\n"

                final_response = StreamResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                logger.error(
                    "deep_research_stream_failed",
                    session_id=session.id,
                    error=str(e),
                    exc_info=True,
                )
                error_response = StreamResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(
            "deep_research_stream_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
