"""VLM request and response models."""

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel


class NIMRequestModel(BaseModel):
    """Request model for VLM inference."""

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
    """Normal response model for VLM inference."""

    chat_id: str
    status: Literal["pending", "running", "completed"]
    message: str | None = None


class ResponseErrorModel(BaseModel):
    """Error response model for VLM inference."""

    chat_id: str
    status: Literal["failed", "timeout"]
    error: str
    message: str | None = None
