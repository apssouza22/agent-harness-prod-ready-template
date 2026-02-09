from datetime import datetime

from pydantic import (
  BaseModel,
  Field,
)


class Token(BaseModel):
  """Token model for authentication.

  Attributes:
      access_token: The JWT access token.
      token_type: The type of token (always "bearer").
      expires_at: The token expiration timestamp.
  """

  access_token: str = Field(..., description="The JWT access token")
  token_type: str = Field(default="bearer", description="The type of token")
  expires_at: datetime = Field(..., description="The token expiration timestamp")


class TokenResponse(BaseModel):
  """Response model for login endpoint.

  Attributes:
      access_token: The JWT access token
      token_type: The type of token (always "bearer")
      expires_at: When the token expires
  """

  access_token: str = Field(..., description="The JWT access token")
  token_type: str = Field(default="bearer", description="The type of token")
  expires_at: datetime = Field(..., description="When the token expires")

