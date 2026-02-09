import re

from pydantic import (
  BaseModel,
  EmailStr,
  Field,
  SecretStr,
  field_validator,
)

from src.app.core.common.token_dtos import Token


class UserResponse(BaseModel):
  """Response model for user operations.

  Attributes:
      id: User's ID
      email: User's email address
      token: Authentication token
  """

  id: int = Field(..., description="User's ID")
  email: str = Field(..., description="User's email address")
  token: Token = Field(..., description="Authentication token")


class UserCreate(BaseModel):
  """Request model for user registration.

  Attributes:
      email: User's email address
      password: User's password
  """

  email: EmailStr = Field(..., description="User's email address")
  password: SecretStr = Field(..., description="User's password", min_length=8, max_length=64)

  @field_validator("password")
  @classmethod
  def validate_password(cls, v: SecretStr) -> SecretStr:
    """Validate password strength.

    Args:
        v: The password to validate

    Returns:
        SecretStr: The validated password

    Raises:
        ValueError: If the password is not strong enough
    """
    password = v.get_secret_value()

    # Check for common password requirements
    if len(password) < 8:
      raise ValueError("Password must be at least 8 characters long")

    if not re.search(r"[A-Z]", password):
      raise ValueError("Password must contain at least one uppercase letter")

    if not re.search(r"[a-z]", password):
      raise ValueError("Password must contain at least one lowercase letter")

    if not re.search(r"[0-9]", password):
      raise ValueError("Password must contain at least one number")

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
      raise ValueError("Password must contain at least one special character")

    return v

