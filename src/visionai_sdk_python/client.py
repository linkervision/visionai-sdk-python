import httpx

from ._base import _BaseClient, _transport_settings
from ._request_attribution import merge_request_attribution
from .auth.credentials import ClientCredentials
from .auth.resource import AuthResource
from .exceptions import AuthenticationError, from_transport_error
from .vlm.resource import VLMResource


class Client(_BaseClient):
    def __init__(
        self,
        auth_url: str,
        vlm_url: str | None = None,
        allowed_issuers: list[str] | None = None,
        verify_ssl: bool | None = None,
        timeout: float | None = None,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        credential: ClientCredentials | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """See ``_BaseClient``. ``http_client`` is an httpx.Client to send through
        (tests, shared pools): its own TLS, timeout and limits apply, so
        ``verify_ssl`` / ``timeout`` / ``max_connections`` /
        ``max_keepalive_connections`` cannot be given with it (``ValueError``);
        the client closes only the one it created."""
        if credential is not None and not isinstance(credential, ClientCredentials):
            raise TypeError(
                "Client takes a ClientCredentials; AsyncClient takes an AsyncClientCredentials"
            )
        settings = _transport_settings(
            http_client,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        super().__init__(
            auth_url=auth_url,
            vlm_url=vlm_url,
            allowed_issuers=allowed_issuers,
            credential=credential,
            **settings,
        )
        self._client = (
            http_client
            if http_client is not None
            else httpx.Client(
                verify=self.verify_ssl,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                ),
            )
        )
        self._http_owner = http_client is None  # a client we built is ours to close
        # Register the different function resource
        self.auth = AuthResource(self)
        self.vlm = VLMResource(self)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute an HTTP request, mapping httpx exceptions to SDK exceptions."""
        merge_request_attribution(kwargs)
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.RequestError as e:
            raise from_transport_error(e) from e
        return self._handle_response(response)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send any request with this client's bearer token.

        ``url`` is a full URL (the client has two bases); ``kwargs`` go to httpx. The
        bearer token goes to whatever host ``url`` names, so ``url`` must come from
        configuration or code, never from a caller's input; redirects are not
        followed unless asked, and httpx drops the header on a cross-origin redirect.

        With a ``credential``, a 401 buys one retry with the credential's current
        token, which it replaces when the rejected one is old enough to have been
        revoked (``REPLACE_COOLDOWN_S``); a 401 on the retry raises. Without a
        credential, or with a body that cannot be sent twice (a stream or a
        file), a 401 raises.

        Raises:
            ValueError: ``url`` is not a full URL.
            AuthenticationError: No token, or the fresh one was refused too.
            RateLimitedError, NetworkError, ServerError, ClientError: As the
                resource methods.
        """
        url = self._full_url(url)
        token = self._ensure_token()
        try:
            return self._request(method, url, **self._with_auth(kwargs, token))
        except AuthenticationError:
            if self._credential is None or not self._resendable(kwargs):
                raise
            self._credential.invalidate(token)
            replacement = self._credential.get_token().token
            if replacement == token:
                raise
        return self._request(method, url, **self._with_auth(kwargs, replacement))

    def close(self) -> None:
        """Close the HTTP client this client created and release its connections."""
        if self._http_owner:
            self._client.close()

    def __enter__(self) -> "Client":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close client."""
        self.close()

    def _refresh_token(self) -> None:
        """Refresh the token with the stored login credentials.

        Client credentials refresh through the credential object instead.

        Raises:
            AuthenticationError: If no credentials are stored or refresh fails.
        """
        if self._credentials is None or self._credentials_type is None:
            raise AuthenticationError("No credentials available for token refresh")

        self.auth.login(
            email=self._credentials["email"],
            password=self._credentials["password"],
        )

    def _ensure_token(self) -> str:
        """The token to send: the credential's, else the stored one, refreshed if expiring.

        Raises:
            AuthenticationError: If no token is available or token expired without credentials.
        """
        if self._credential is not None:
            return self._credential.get_token().token
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
        return self._access_token
