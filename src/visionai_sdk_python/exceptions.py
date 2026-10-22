import httpx


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


class RateLimitedError(ClientError):
    """429 - Too many requests; on client-token, Platform is rate limiting mints."""

    def __init__(self, message: str) -> None:
        super().__init__(429, message)


class ServerError(APIError):
    """5xx - Server-side failure, consider retry."""


class NetworkError(VisionaiSDKError):
    """Connection or timeout failure."""


class JwksDiscoveryError(VisionaiSDKError):
    """Failed to fetch or parse the OIDC discovery document or JWKS endpoint."""


def from_transport_error(error: httpx.RequestError) -> VisionaiSDKError:
    """The SDK exception for an httpx request that never got a response."""
    if isinstance(error, httpx.TimeoutException):
        return NetworkError("Request timed out")
    if isinstance(error, httpx.NetworkError):
        return NetworkError(f"Network error: {error}")
    return VisionaiSDKError(f"Request failed: {error}")


def raise_for_status(response: httpx.Response) -> httpx.Response:
    """Raise SDK-specific exceptions for non-2xx responses.

    Raises:
        AuthenticationError: If the server returns 401
        PermissionDeniedError: If the server returns 403
        ClientError: If the server returns any other 4xx
        ServerError: If the server returns 5xx
        APIError: If the server returns any other non-2xx status
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = None
        if e.response.content:
            try:
                body = e.response.json()
            except Exception:
                pass
        if isinstance(body, dict):
            detail: str = body.get("detail") or body.get("message") or str(e)
        else:
            detail = str(e)
        status = e.response.status_code
        if status == 401:
            raise AuthenticationError(detail) from e
        if status == 403:
            raise PermissionDeniedError(detail) from e
        if status == 429:
            raise RateLimitedError(detail) from e
        if 400 <= status < 500:
            raise ClientError(status, detail) from e
        if 500 <= status < 600:
            raise ServerError(status, detail) from e
        raise APIError(status, detail) from e
    return response
