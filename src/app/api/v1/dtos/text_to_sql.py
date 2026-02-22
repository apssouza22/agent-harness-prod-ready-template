"""Request and response models for the Text-to-SQL API."""

from typing import List

from pydantic import BaseModel, Field

from src.app.core.common.model.message import Message


class TextSQLRequest(BaseModel):
    """Request model for text-to-SQL endpoint.

    Attributes:
        query: Natural language question to translate into SQL and execute.
    """

    query: str = Field(
        ...,
        description="Natural language question to translate into SQL and execute",
        min_length=1,
        max_length=2000,
    )


class TextSQLResponse(BaseModel):
    """Response model for text-to-SQL endpoint.

    Attributes:
        messages: List of messages from the agent's response.
    """

    messages: List[Message] = Field(..., description="Agent response messages")
