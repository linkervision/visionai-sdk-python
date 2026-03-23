class VisionaiSDKError(Exception):
    """Base exception for all SDK errors."""


class AuthenticationError(VisionaiSDKError):
    """401 - Invalid credentials/password."""


class PermissionDeniedError(VisionaiSDKError):
    """403 - Insufficient permissions."""


class ClientError(VisionaiSDKError):
    """4xx (excluding 401/403) - Caller-side error."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"[{status_code}] {message}")


class ServerError(VisionaiSDKError):
    """5xx - Server-side failure, consider retry."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"[{status_code}] {message}")


class NetworkError(VisionaiSDKError):
    """Connection or timeout failure."""
