from pydantic import AnyHttpUrl, BaseModel, Field
from typing import Literal

class TokenResponse(BaseModel):
    """Response model for authentication token endpoints.

    Used by both /api/users/jwt and /api/users/client-token endpoints.
    """
    access_token: str = Field(..., description="JWT access token")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    token_type: str = Field(..., description="Token type (e.g., 'Bearer')")


class NIMRequestModel(BaseModel):
    img: str | list[str]
    prompt: str
    temperature: float | None = 0.2
    max_tokens: int | None = 500
    top_p: float | None = 0.7
    stream: bool = False
    use_cache: bool = True
    num_beams: int | None = 1
    api_endpoint: AnyHttpUrl | None = None
    hook: AnyHttpUrl | None = None
    # Will only work within hook (will not validated though)
    use_response_postprocess: bool = False


class ResponseNormalModel(BaseModel):
    chat_id: str
    status: Literal["pending", "running", "completed"]
    message: str | None = None


class ResponseErrorModel(BaseModel):
    chat_id: str
    status: Literal["failed", "timeout"]
    error: str
    message: str | None = None
