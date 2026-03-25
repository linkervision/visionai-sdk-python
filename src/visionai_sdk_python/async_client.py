import httpx

from ._base import _BaseClient
from .exceptions import NetworkError, VisionaiSDKError
from .models import TokenResponse

class AsyncClient(_BaseClient):
    def __init__(
        self,
        auth_url: str,
        vlm_url: str,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        super().__init__(
            auth_url=auth_url,
            vlm_url=vlm_url,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._client = httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
        )


    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        await self._client.aclose()


    async def __aenter__(self) -> "AsyncClient":
        """Async context manager entry."""
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - close client."""
        await self.close()


    async def get_access_token(self, client_id: str, client_secret: str) -> TokenResponse:
        """Get access token using client credentials flow.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret

        Returns:
            TokenResponse with access_token, expires_in, and token_type

        Raises:
            ValueError: If client_id or client_secret is empty
            NetworkError: If the request times out or a network error occurs
            VisionaiSDKError: If the request fails for any other reason
        """
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if not client_secret.strip():
            raise ValueError("client_secret must not be empty")

        try:
            response = await self._client.post(
                self._build_url(self.auth_url, "/api/users/client-token"),
                json={
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
        except httpx.TimeoutException as e:
            raise NetworkError("Request timed out") from e
        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}") from e
        except httpx.RequestError as e:
            raise VisionaiSDKError(f"Request failed: {e}") from e
        self._handle_response(response)
        return TokenResponse(**response.json())


    async def login(self, email: str, password: str) -> TokenResponse:
        """Login with email and password to get JWT token.

        Args:
            email: User email address
            password: User password

        Returns:
            TokenResponse with access_token, expires_in, and token_type

        Raises:
            ValueError: If email or password is empty
            NetworkError: If the request times out or a network error occurs
            VisionaiSDKError: If the request fails for any other reason
        """
        if not email.strip():
            raise ValueError("email must not be empty")
        if not password.strip():
            raise ValueError("password must not be empty")

        try:
            response = await self._client.post(
                self._build_url(self.auth_url, "/api/users/jwt"),
                json={
                    "email": email,
                    "password": password,
                },
            )
        except httpx.TimeoutException as e:
            raise NetworkError("Request timed out") from e
        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}") from e
        except httpx.RequestError as e:
            raise VisionaiSDKError(f"Request failed: {e}") from e
        self._handle_response(response)
        return TokenResponse(**response.json())
