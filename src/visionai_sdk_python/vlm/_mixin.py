"""Shared VLM business logic for sync and async resources."""

from .models import NIMRequestModel, ResponseErrorModel, ResponseNormalModel


class VLMMixin:
    """Shared VLM logic (validation, data preparation, parsing).

    This mixin contains all business logic that doesn't involve I/O operations.
    Sync and async resources inherit from this to avoid code duplication.
    """

    # ==========================================================================
    # Request preparation methods
    # ==========================================================================

    def _prepare_chat_request(self, payload: NIMRequestModel | dict) -> dict:
        """Prepare chat request payload.

        Args:
            payload: Chat request as NIMRequestModel instance or dict

        Returns:
            JSON-serializable dict for API request
        """
        nim_request = (
            NIMRequestModel.model_validate(payload)
            if isinstance(payload, dict)
            else payload
        )
        return nim_request.model_dump(mode="json")

    # ==========================================================================
    # Response parsing methods
    # ==========================================================================

    def _parse_chat_response(
        self, data: dict
    ) -> ResponseNormalModel | ResponseErrorModel:
        """Parse chat response from API.

        Args:
            data: Response data dict

        Returns:
            ResponseNormalModel for success, ResponseErrorModel for failures
        """
        if data.get("status") in ("failed", "timeout"):
            return ResponseErrorModel(**data)
        return ResponseNormalModel(**data)
