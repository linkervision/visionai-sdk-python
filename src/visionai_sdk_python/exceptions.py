class VisionaiSDKError(Exception):
    """Base exception for all SDK errors."""


class APIError(VisionaiSDKError):
    """Base exception for all HTTP API errors (has status_code)."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"[{status_code}] {message}")


class ClientError(APIError):
    """4xx - Client-side error."""


class AuthenticationError(ClientError):
    """401 - Invalid credentials/password."""

    def __init__(self, message: str) -> None:
        super().__init__(401, message)


class PermissionDeniedError(ClientError):
    """403 - Insufficient permissions."""

    def __init__(self, message: str) -> None:
        super().__init__(403, message)


class ServerError(APIError):
    """5xx - Server-side failure, consider retry."""


class NetworkError(VisionaiSDKError):
    """Connection or timeout failure."""
