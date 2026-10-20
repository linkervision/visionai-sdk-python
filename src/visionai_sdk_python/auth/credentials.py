"""One client-credentials token for every client and caller in a process.

Platform mints at most 50 tokens per client per month, so the token is cached
until five minutes before it expires, one mint runs at a time, a 401 buys one
replacement, and a 429 stops minting for an hour.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .._base import _BaseClient
from .._request_attribution import merge_request_attribution
from ..endpoints import AuthEndpoint
from ..exceptions import RateLimitedError, VisionaiSDKError
from .models import TokenResponse

logger = logging.getLogger(__name__)

# A token counts as expired this long before Platform says so.
REFRESH_BUFFER_S = 300
# Anything shorter would mint on every call; Platform issues 3-day tokens.
MIN_EXPIRES_IN_S = 2 * REFRESH_BUFFER_S
# After a 429 (the monthly quota is used up) no mint is tried for this long.
QUOTA_COOLDOWN_S = 3600

_now = time.monotonic


@dataclass(frozen=True)
class AccessToken:
    """A bearer token and the ``time.monotonic()`` second it expires at."""

    token: str
    expires_on: float


class _Credentials:
    """What the sync and async variants share: the cache, the 401 and 429 rules."""

    def __init__(self, auth_url: str, client_id: str, client_secret: str) -> None:
        if not auth_url.strip():
            raise ValueError("auth_url must not be empty")
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if not client_secret.strip():
            raise ValueError("client_secret must not be empty")
        self.auth_url = auth_url.strip()
        self.client_id = client_id.strip()
        self._client_secret = client_secret
        self._url = _BaseClient._build_url(self.auth_url, AuthEndpoint.CLIENT_TOKEN)
        self._token: AccessToken | None = None
        #: The last token endpoint response, for callers that want ``token_type`` too.
        self.last_response: TokenResponse | None = None
        # True while the cached token is the replacement a 401 bought.
        self._replaced = False
        self._blocked_until = 0.0

    def _cached(self) -> AccessToken | None:
        if self._token and _now() < self._token.expires_on - REFRESH_BUFFER_S:
            return self._token
        return None

    def _mint_request(self) -> dict[str, Any]:
        """kwargs for the client-token POST; refuses while a 429 cooldown runs."""
        if _now() < self._blocked_until:
            raise RateLimitedError(
                "client-token quota used up; not minting until the cooldown ends"
            )
        if self._token is not None:
            # Natural refresh: the next 401 may buy one replacement again.
            self._replaced = False
        kwargs: dict[str, Any] = {
            "json": {"client_id": self.client_id, "client_secret": self._client_secret}
        }
        merge_request_attribution(kwargs)
        return kwargs

    def _store(self, response: httpx.Response) -> AccessToken:
        try:
            _BaseClient._handle_response(response)
        except RateLimitedError:
            self._blocked_until = _now() + QUOTA_COOLDOWN_S
            raise
        try:
            parsed = TokenResponse(**response.json())
        except (ValueError, TypeError) as e:
            raise VisionaiSDKError(f"client-token response is not a token: {e}") from e
        self.last_response = parsed
        self._token = AccessToken(
            parsed.access_token, _now() + max(parsed.expires_in, MIN_EXPIRES_IN_S)
        )
        logger.info(
            "platform token minted",
            extra={"client_id": self.client_id, "expires_in": parsed.expires_in},
        )
        return self._token

    def invalidate(self, token: str) -> bool:
        """Report a 401 for ``token``. True: dropped, so the next ``get_token()`` mints.

        A 401 for a token already replaced changes nothing. A 401 for the
        replacement itself is not expiry but a client or allowlist problem that
        another mint would not fix, so that token stays until its own refresh.
        """
        if self._token is None or token != self._token.token:
            return False
        if self._replaced:
            logger.error(
                "replacement token rejected too; keeping it until it expires",
                extra={"client_id": self.client_id},
            )
            return False
        self._token = None
        self._replaced = True
        return True


class ClientCredentials(_Credentials):
    """Client-credentials token for one Platform client, for sync code.

    Pass it to ``Client(credential=...)`` and use ``get_token()`` for requests the
    SDK does not make itself. One instance serves any number of clients and callers.
    """

    def __init__(
        self,
        auth_url: str,
        client_id: str,
        client_secret: str,
        *,
        http: httpx.Client | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(auth_url, client_id, client_secret)
        self._http = http or httpx.Client(verify=verify_ssl, timeout=timeout)
        self._owns_http = http is None
        self._lock = threading.Lock()

    def get_token(self) -> AccessToken:
        """The cached token, or a fresh one when it is within five minutes of expiry.

        Raises:
            AuthenticationError: Platform refused the client credentials (401).
            RateLimitedError: The monthly mint quota is used up (429), or a 429 was
                seen less than an hour ago.
            NetworkError: Platform could not be reached.
            VisionaiSDKError: Any other failure.
        """
        with self._lock:
            cached = self._cached()
            if cached is not None:
                return cached
            kwargs = self._mint_request()
            try:
                response = self._http.post(self._url, **kwargs)
            except httpx.RequestError as e:
                raise _BaseClient._map_transport_error(e) from e
            return self._store(response)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> ClientCredentials:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AsyncClientCredentials(_Credentials):
    """Client-credentials token for one Platform client, for asyncio code.

    Pass it to ``AsyncClient(credential=...)`` and use ``await get_token()`` for
    requests the SDK does not make itself, such as a WebSocket upgrade. One
    instance serves any number of clients and callers.
    """

    def __init__(
        self,
        auth_url: str,
        client_id: str,
        client_secret: str,
        *,
        http: httpx.AsyncClient | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(auth_url, client_id, client_secret)
        self._http = http or httpx.AsyncClient(verify=verify_ssl, timeout=timeout)
        self._owns_http = http is None
        self._lock = asyncio.Lock()

    async def get_token(self) -> AccessToken:
        """The cached token, or a fresh one when it is within five minutes of expiry.

        Concurrent callers wait for one mint. Raises as ``ClientCredentials.get_token``.
        """
        async with self._lock:
            cached = self._cached()
            if cached is not None:
                return cached
            kwargs = self._mint_request()
            try:
                response = await self._http.post(self._url, **kwargs)
            except httpx.RequestError as e:
                raise _BaseClient._map_transport_error(e) from e
            return self._store(response)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClientCredentials:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
