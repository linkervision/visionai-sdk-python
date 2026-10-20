import httpx

from ._base import _BaseClient
from ._request_attribution import merge_request_attribution
from .auth.credentials import ClientCredentials
from .auth.resource import AuthResource
from .exceptions import AuthenticationError
from .vlm.resource import VLMResource


class Client(_BaseClient):
    def __init__(
        self,
        auth_url: str,
        vlm_url: str | None = None,
        allowed_issuers: list[str] | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        credential: ClientCredentials | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """See ``_BaseClient``; ``http_client`` is an httpx.Client to send through
        (tests, shared pools); the client closes only the one it created."""
        super().__init__(
            auth_url=auth_url,
            vlm_url=vlm_url,
            allowed_issuers=allowed_issuers,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            credential=credential,
        )
        self._client = http_client or httpx.Client(
            verify=self.verify_ssl,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
        )
        self._owns_http = http_client is None
        # Register the different function resource
        self.auth = AuthResource(self)
        self.vlm = VLMResource(self)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute an HTTP request, mapping httpx exceptions to SDK exceptions."""
        merge_request_attribution(kwargs)
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.RequestError as e:
            raise self._map_transport_error(e) from e
        return self._handle_response(response)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send any request with this client's bearer token.

        ``url`` is absolute, or a path under ``auth_url``; ``kwargs`` go to httpx.
        A 401 buys one fresh token (from ``credential`` or the stored login) and
        one retry, so a token revoked server-side costs one extra request.

        Raises:
            AuthenticationError: No token, or the fresh one was refused too.
            RateLimitedError, NetworkError, ServerError, ClientError: As the
                resource methods.
        """
        url = self._absolute(url)
        self._ensure_token()
        token = self._access_token
        try:
            return self._request(method, url, **self._with_auth(kwargs, token))
        except AuthenticationError:
            if not self._renew_rejected(token):
                raise
        return self._request(method, url, **self._with_auth(kwargs, self._access_token))

    def _renew_rejected(self, token: str) -> bool:
        """A 401 for ``token``: get a different one, once; False when none would help."""
        if self._credential is not None:
            if not self._credential.invalidate(token):
                return False
            self._access_token = self._credential.get_token().token
            return True
        if self._credentials is None or self._access_token != token:
            return False
        self._refresh_token()
        return self._access_token != token

    def close(self) -> None:
        """Close the HTTP client this client created and release its connections."""
        if self._owns_http:
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
        if self._credential is not None:
            self._access_token = (self._credential.get_token()).token
            return
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
