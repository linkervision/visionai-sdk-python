import httpx

from ._base import _BaseClient
from ._request_attribution import merge_request_attribution
from .auth.async_resource import AsyncAuthResource
from .auth.credentials import AsyncClientCredentials
from .exceptions import AuthenticationError, from_transport_error
from .vlm.async_resource import AsyncVLMResource


class AsyncClient(_BaseClient):
    def __init__(
        self,
        auth_url: str,
        vlm_url: str | None = None,
        allowed_issuers: list[str] | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        credential: AsyncClientCredentials | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """See ``_BaseClient``; ``http_client`` is an httpx.AsyncClient to send through
        (tests, shared pools); the client closes only the one it created."""
        if credential is not None and not isinstance(
            credential, AsyncClientCredentials
        ):
            raise TypeError(
                "AsyncClient takes an AsyncClientCredentials; Client takes a ClientCredentials"
            )
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
        self._client = http_client or httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
        )
        self._http_owner = http_client is None  # a client we built is ours to close
        # Register the different function resource
        self.auth = AsyncAuthResource(self)
        self.vlm = AsyncVLMResource(self)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute an async HTTP request, mapping httpx exceptions to SDK exceptions."""
        merge_request_attribution(kwargs)
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.RequestError as e:
            raise from_transport_error(e) from e
        return self._handle_response(response)

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send any request with this client's bearer token.

        ``url`` is a full URL (the client has two bases); ``kwargs`` go to httpx.
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
        token = await self._ensure_token()
        try:
            return await self._request(method, url, **self._with_auth(kwargs, token))
        except AuthenticationError:
            if self._credential is None or not self._resendable(kwargs):
                raise
            self._credential.invalidate(token)
            replacement = (await self._credential.get_token()).token
            if replacement == token:
                raise
        return await self._request(method, url, **self._with_auth(kwargs, replacement))

    async def close(self) -> None:
        """Close the HTTP client this client created and release its connections."""
        if self._http_owner:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - close client."""
        await self.close()

    async def _refresh_token(self) -> None:
        """Refresh the token with the stored login credentials.

        Client credentials refresh through the credential object instead.

        Raises:
            AuthenticationError: If no credentials are stored or refresh fails.
        """
        if self._credentials is None or self._credentials_type is None:
            raise AuthenticationError("No credentials available for token refresh")

        await self.auth.login(
            email=self._credentials["email"],
            password=self._credentials["password"],
        )

    async def _ensure_token(self) -> str:
        """The token to send: the credential's, else the stored one, refreshed if expiring.

        Raises:
            AuthenticationError: If no token is available or token expired without credentials.
        """
        if self._credential is not None:
            return (await self._credential.get_token()).token
        if self._access_token is None:
            raise AuthenticationError(
                "Not authenticated. Call login() or get_access_token() first."
            )

        if self._is_token_expiring_soon():
            if self._credentials is None:
                raise AuthenticationError(
                    "Token expired and no credentials available for refresh"
                )
            await self._refresh_token()
        return self._access_token
