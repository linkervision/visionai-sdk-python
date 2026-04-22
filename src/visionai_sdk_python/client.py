import logging
import httpx

from ._base import _BaseClient
from .exceptions import AuthenticationError, NetworkError, VisionaiSDKError
from .auth.resource import AuthResource
from .vlm.resource import VLMResource


class Client(_BaseClient):
    def __init__(
        self,
        auth_url: str,
        vlm_url: str,
        allowed_issuers: list[str] | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        super().__init__(
            auth_url=auth_url,
            vlm_url=vlm_url,
            allowed_issuers=allowed_issuers,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._client = httpx.Client(
            verify=self.verify_ssl,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
        )
        # Register the different function resource
        self.auth = AuthResource(self)
        self.vlm = VLMResource(self)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute an HTTP request, mapping httpx exceptions to SDK exceptions."""
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as e:
            raise NetworkError("Request timed out") from e
        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}") from e
        except httpx.RequestError as e:
            raise VisionaiSDKError(f"Request failed: {e}") from e
        return self._handle_response(response)

    def close(self) -> None:
        """Close the HTTP client and release connections."""
        self._client.close()

    def __enter__(self) -> "Client":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close client."""
        self.close()

    def _refresh_token(self) -> None:
        """Refresh token using stored credentials.

        Raises:
            AuthenticationError: If no credentials are stored or refresh fails.
        """
        if self._credentials is None or self._credentials_type is None:
            raise AuthenticationError("No credentials available for token refresh")

        if self._credentials_type == "login":
            self.auth.login(
                email=self._credentials["email"],
                password=self._credentials["password"],
            )
        elif self._credentials_type == "client":
            self.auth.get_access_token(
                client_id=self._credentials["client_id"],
                client_secret=self._credentials["client_secret"],
            )

    def _ensure_token(self) -> None:
        """Ensure a valid token is available, refreshing if necessary.

        Raises:
            AuthenticationError: If no token is available or token expired without credentials.
        """
        if self._access_token is None:
            raise AuthenticationError(
                "Not authenticated. Call login() or get_access_token() first."
            )

        if self._is_token_expiring_soon():
            if self._credentials is None:
                raise AuthenticationError(
                    "Token expired and no credentials available for refresh"
                )
            self._refresh_token()
