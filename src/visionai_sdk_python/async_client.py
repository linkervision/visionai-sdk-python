import logging
import httpx
import jwt

from ._base import _BaseClient
from .endpoints import AuthEndpoint, VLMEndpoint
from .models import TokenResponse
from .exceptions import AuthenticationError, JwksDiscoveryError, NetworkError, VisionaiSDKError
from .models import TokenResponse, NIMRequestModel, ResponseNormalModel, ResponseErrorModel


logger = logging.getLogger(__name__)


class AsyncClient(_BaseClient):
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
        self._client = httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
        )


    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute an async HTTP request, mapping httpx exceptions to SDK exceptions."""
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as e:
            raise NetworkError("Request timed out") from e
        except httpx.NetworkError as e:
            raise NetworkError(f"Network error: {e}") from e
        except httpx.RequestError as e:
            raise VisionaiSDKError(f"Request failed: {e}") from e
        return self._handle_response(response)

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

        The token is stored internally and will be used automatically for subsequent
        API calls. If the token expires, it will be automatically refreshed.

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

        response = await self._request(
            "POST",
            self._build_url(self.auth_url, AuthEndpoint.CLIENT_TOKEN),
            json={"client_id": client_id, "client_secret": client_secret},
        )
        token_response = TokenResponse(**response.json())

        # Store token and credentials for auto-refresh
        self._store_token(
            access_token=token_response.access_token,
            expires_in=token_response.expires_in,
            credentials={"client_id": client_id, "client_secret": client_secret},
            credentials_type="client",
        )

        return token_response


    async def login(self, email: str, password: str) -> TokenResponse:
        """Login with email and password to get JWT token.

        The token is stored internally and will be used automatically for subsequent
        API calls. If the token expires, it will be automatically refreshed.

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

        response = await self._request(
            "POST",
            self._build_url(self.auth_url, AuthEndpoint.LOGIN),
            json={"email": email, "password": password},
        )
        token_response = TokenResponse(**response.json())

        # Store token and credentials for auto-refresh
        self._store_token(
            access_token=token_response.access_token,
            expires_in=token_response.expires_in,
            credentials={"email": email, "password": password},
            credentials_type="login",
        )

        return token_response

    async def _refresh_token(self) -> None:
        """Refresh token using stored credentials.

        Raises:
            AuthenticationError: If no credentials are stored or refresh fails.
        """
        if self._credentials is None or self._credentials_type is None:
            raise AuthenticationError("No credentials available for token refresh")

        if self._credentials_type == "login":
            await self.login(
                email=self._credentials["email"],
                password=self._credentials["password"],
            )
        elif self._credentials_type == "client":
            await self.get_access_token(
                client_id=self._credentials["client_id"],
                client_secret=self._credentials["client_secret"],
            )

    async def _ensure_token(self) -> None:
        """Ensure a valid token is available, refreshing if necessary.

        Raises:
            AuthenticationError: If no token is available or token expired without credentials.
        """
        if self._access_token is None:
            raise AuthenticationError("Not authenticated. Call login() or get_access_token() first.")

        if self._is_token_expiring_soon():
            if self._credentials is None:
                raise AuthenticationError("Token expired and no credentials available for refresh")
            await self._refresh_token()

    async def is_token_valid(self, access_token: str) -> bool:
        """Check whether a JWT access token is currently valid.

        Validates the token's signature and expiration without raising exceptions.

        Args:
            access_token: JWT access token to validate.

        Returns:
            ``True`` if the token passes signature and expiration checks,
            ``False`` otherwise (invalid token or JWKS service unavailable).

        Note:
            Logs token validation failures. Unexpected errors will propagate
            to allow fail-fast behavior for programming errors.
        """
        try:
            await self._jwt_verifier.verify_async(access_token)
            return True
        except jwt.InvalidTokenError as e:
            # Expected: expired, malformed, invalid signature, missing claims
            logger.warning(
                "%s: Token validation failed",
                type(e).__name__,
                extra={"jwt_error_type": type(e).__name__, "jwt_error_message": str(e)}
            )
            return False
        except jwt.PyJWKClientError as e:
            # Expected: JWKS endpoint unavailable, network issues
            logger.error(
                "%s: JWKS client error during token validation",
                type(e).__name__,
                extra={"jwt_error_type": "PyJWKClientError", "jwt_error_message": str(e)}
            )
            return False
        except JwksDiscoveryError as e:
            # Expected: OIDC discovery endpoint unreachable or returned unexpected response
            logger.error(
                "%s: OIDC discovery failed during token validation",
                type(e).__name__,
                extra={"jwt_error_type": type(e).__name__, "jwt_error_message": str(e)}
            )
            return False

    async def chat(
        self,
        payload: NIMRequestModel | dict,
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
        await self._ensure_token()

        nim_request = (
            NIMRequestModel.model_validate(payload)
            if isinstance(payload, dict)
            else payload
        )
        response = await self._request(
            "POST",
            self._build_url(self.vlm_url, VLMEndpoint.CHAT),
            headers=self._build_auth_header(self._access_token),
            json=nim_request.model_dump(mode="json"),
        )
        data = response.json()
        if data.get("status") in ("failed", "timeout"):
            return ResponseErrorModel(**data)
        return ResponseNormalModel(**data)


    async def get_chat(self, result_id: str) -> ResponseNormalModel | ResponseErrorModel:
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
        await self._ensure_token()

        response = await self._request(
            "GET",
            self._build_url(self.vlm_url, f"{VLMEndpoint.CHAT}/{result_id}"),
            headers=self._build_auth_header(self._access_token),
        )
        data = response.json()
        if data.get("status") in ("failed", "timeout"):
            return ResponseErrorModel(**data)
        return ResponseNormalModel(**data)
