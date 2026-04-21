from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    """Response model for authentication token endpoints.

    Used by both /api/users/jwt and /api/users/client-token endpoints.
    """
    access_token: str = Field(..., description="JWT access token")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    token_type: str = Field(..., description="Token type (e.g., 'Bearer')")
