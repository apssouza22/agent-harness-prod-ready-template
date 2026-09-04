"""Text-to-SQL API endpoints for natural language database queries.

This module provides endpoints for translating natural language questions
into SQL queries and executing them against the database.
"""

from fastapi import APIRouter, HTTPException, Request

from src.app.api.security.limiter import limiter
from src.app.api.v1.dtos.text_to_sql import TextSQLRequest, TextSQLResponse
from src.app.core.common.config import settings
from src.app.core.common.logging import logger
from src.app.core.common.model.message import Message
from src.app.dependencies import CurrentSessionDep, TextToSqlAgentDep

router = APIRouter()


@router.post("/query", response_model=TextSQLResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["text_to_sql"][0])
async def text_to_sql_query(
    request: Request,
    sql_request: TextSQLRequest,
    session: CurrentSessionDep,
    agent: TextToSqlAgentDep,
):
    """Submit a natural language query to be translated into SQL and executed."""
    try:
        logger.info(
            "text_to_sql_request_received",
            session_id=session.id,
            query_length=len(sql_request.query),
        )

        messages = [Message(role="user", content=sql_request.query)]
        result = await agent.agent_invoke(messages, session.id, user_id=session.user_id)

        logger.info("text_to_sql_request_processed", session_id=session.id)
        return TextSQLResponse(messages=result)
    except Exception as e:
        logger.error(
            "text_to_sql_request_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
