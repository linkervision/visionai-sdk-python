"""VLM (Vision Language Model) resource for sync operations."""

from typing import TYPE_CHECKING

from ..endpoints import VLMEndpoint
from ._mixin import VLMMixin
from .models import NIMRequestModel, ResponseErrorModel, ResponseNormalModel

if TYPE_CHECKING:
    from ..client import Client


class VLMResource(VLMMixin):
    """Synchronous VLM operations."""

    _sdk_client: "Client"

    def __init__(self, client: "Client") -> None:
        """Initialize VLM resource.

        Args:
            client: Parent sync client instance
        """
        self._sdk_client = client

    def chat(
        self, payload: NIMRequestModel | dict
    ) -> ResponseNormalModel | ResponseErrorModel:
        """Submit an inference request to the VLM service.

        Sent with the client's token (``credential``, ``login()`` or ``get_access_token()``);
        a 401 gets one fresh token and one retry.

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
        # Prepare request (from Mixin)
        body = self._prepare_chat_request(payload)

        # I/O operation (sync); the client adds the token and retries once on 401
        response = self._sdk_client.request(
            "POST",
            self._sdk_client._build_url(self._sdk_client._vlm_base(), VLMEndpoint.CHAT),
            json=body,
        )

        # Parse response (from Mixin)
        return self._parse_chat_response(response.json())

    def get_chat(self, result_id: str) -> ResponseNormalModel | ResponseErrorModel:
        """Poll the result of a previously submitted inference request.

        Sent with the client's token (``credential``, ``login()`` or ``get_access_token()``);
        a 401 gets one fresh token and one retry.

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
        # I/O operation (sync); the client adds the token and retries once on 401
        response = self._sdk_client.request(
            "GET",
            self._sdk_client._build_url(
                self._sdk_client._vlm_base(), f"{VLMEndpoint.CHAT}/{result_id}"
            ),
        )

        # Parse response (from Mixin)
        return self._parse_chat_response(response.json())
