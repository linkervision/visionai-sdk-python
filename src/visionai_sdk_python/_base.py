import httpx

from .exceptions import (
    APIError,
    AuthenticationError,
    ClientError,
    PermissionDeniedError,
    ServerError,
)
from ._jwt_verifier import JwtVerifier


class _BaseClient:
    """Base class for VisionAI SDK clients.

    Holds shared connection configuration and provides
    common URL/header builder utilities.
    """

    def __init__(
        self,
        auth_url: str,
        vlm_url: str,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        """Initialize the client with connection settings.

        Args:
            auth_url: Base URL for the authentication service.
            vlm_url: Base URL for the VLM inference service.
            verify_ssl: Whether to verify TLS certificates.
            timeout: Default request timeout in seconds.
            max_connections: Maximum number of concurrent connections in the pool.
            max_keepalive_connections: Maximum number of idle keep-alive connections.
        """
        if not auth_url.strip():
            raise ValueError("auth_url must not be empty")
        if not vlm_url.strip():
            raise ValueError("vlm_url must not be empty")
        self.auth_url = auth_url.strip()
        self.vlm_url = vlm_url.strip()
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self._jwt_verifier = JwtVerifier(verify_ssl=verify_ssl, timeout=timeout)

    @staticmethod
    def _build_url(base_url: str, path: str) -> str:
        """Join a base URL and a path, normalizing slashes."""
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _build_auth_header(access_token: str) -> dict[str, str]:
        """Build authorization header."""
        if not access_token:
            raise ValueError("access_token must not be empty")
        return {"Authorization": f"Bearer {access_token}"}


    @staticmethod
    def _handle_response(response: httpx.Response) -> httpx.Response:
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
            if 400 <= status < 500:
                raise ClientError(status, detail) from e
            if 500 <= status < 600:
                raise ServerError(status, detail) from e
            raise APIError(status, detail)
        return response
