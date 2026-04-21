"""VLM (Vision Language Model) resource for async operations."""

from typing import TYPE_CHECKING

from ..endpoints import VLMEndpoint
from .models import NIMRequestModel, ResponseErrorModel, ResponseNormalModel
from ._mixin import VLMMixin

if TYPE_CHECKING:
    from ..async_client import AsyncClient


class AsyncVLMResource(VLMMixin):
    """Asynchronous VLM operations."""

    def __init__(self, client: "AsyncClient") -> None:
        """Initialize async VLM resource.

        Args:
            client: Parent async client instance
        """
        self._client = client

    async def chat(
        self, payload: NIMRequestModel | dict
    ) -> ResponseNormalModel | ResponseErrorModel:
        """Submit an inference request to the VLM service.

        Uses the internally stored access token obtained from login() or get_access_token().
        If the token is expiring soon, it will be automatically refreshed.

        Args:
            payload: Inference parameters as a NIMRequestModel instance or a dict
                whose keys match NIMRequestModel fields (validated via model_validate).

        Returns:
            ResponseNormalModel if the request is accepted (status: pending/running/completed),
            or ResponseErrorModel if the inference failed or timed out.

        Raises:
            ValidationError: If payload is a dict that fails NIMRequestModel validation.
            AuthenticationError: If not authenticated or token expired without refresh credentials.
            NetworkError: If the request times out or a network error occurs.
            VisionaiSDKError: If the request fails for any other reason.
        """
        # Ensure token is valid
        await self._client._ensure_token()

        # Prepare request (from Mixin)
        body = self._prepare_chat_request(payload)

        # I/O operation (async)
        response = await self._client._request(
            "POST",
            self._client._build_url(self._client.vlm_url, VLMEndpoint.CHAT),
            headers=self._client._build_auth_header(self._client._access_token),
            json=body,
        )

        # Parse response (from Mixin)
        return self._parse_chat_response(response.json())

    async def get_chat(
        self, result_id: str
    ) -> ResponseNormalModel | ResponseErrorModel:
        """Poll the result of a previously submitted inference request.

        Uses the internally stored access token obtained from login() or get_access_token().
        If the token is expiring soon, it will be automatically refreshed.

        Args:
            result_id: Chat result ID returned from a prior chat() call.

        Returns:
            ResponseNormalModel if the result is available (status: pending/running/completed),
            or ResponseErrorModel if the inference failed or timed out.

        Raises:
            AuthenticationError: If not authenticated or token expired without refresh credentials.
            NetworkError: If the request times out or a network error occurs.
            VisionaiSDKError: If the request fails for any other reason.
        """
        # Ensure token is valid
        await self._client._ensure_token()

        # I/O operation (async)
        response = await self._client._request(
            "GET",
            self._client._build_url(
                self._client.vlm_url, f"{VLMEndpoint.CHAT}/{result_id}"
            ),
            headers=self._client._build_auth_header(self._client._access_token),
        )

        # Parse response (from Mixin)
        return self._parse_chat_response(response.json())
