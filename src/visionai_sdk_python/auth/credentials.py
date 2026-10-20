"""One client-credentials token for every client and caller in a process.

A mint is a round trip to Platform and, on sites that cap client tokens, a
scarce one, so the token is cached until shortly before it expires, one mint
runs at a time, a refresh that fails keeps the current token while Platform
still accepts it, a 401 replaces a token that is old enough to have been
revoked, and a 429 stops minting for an hour.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .._base import _BaseClient
from .._request_attribution import merge_request_attribution
from ..endpoints import AuthEndpoint
from ..exceptions import (
    RateLimitedError,
    VisionaiSDKError,
    from_transport_error,
    raise_for_status,
)
from .models import TokenResponse

logger = logging.getLogger(__name__)

# A token is refreshed this long before it expires, or at half its lifetime when
# that is sooner, so a token is never served after Platform stops accepting it.
REFRESH_BUFFER_S = 300
# After a 429 no mint is tried for this long.
RATE_LIMIT_COOLDOWN_S = 3600
# A 401 on a token minted less than this long ago does not replace it.
REPLACE_COOLDOWN_S = 30 * 60

_now = time.monotonic


@dataclass(frozen=True)
class AccessToken:
    """A bearer token with the ``time.monotonic()`` seconds it expires at and was issued."""

    token: str
    expires_on: float
    token_type: str
    issued_at: float

    def expires_in(self) -> int:
        """Seconds left before ``expires_on``, never negative."""
        return max(int(self.expires_on - _now()), 0)


class _Credentials:
    """What the sync and async variants share: the cache, the refresh, 401 and 429 rules."""

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
        self._current: AccessToken | None = None  # the token this credential holds
        self._blocked_until = 0.0
        self._minting = False  # a mint is in flight (set under the lock)

    def is_for(self, client_id: str, client_secret: str) -> bool:
        """True when this credential was built for exactly these client credentials."""
        return self.client_id == client_id.strip() and hmac.compare_digest(
            self._client_secret, client_secret
        )

    def _fresh(self) -> AccessToken | None:
        """The current token while it is not yet due for a refresh."""
        token = self._current
        if token is None:
            return None
        buffer = min(REFRESH_BUFFER_S, (token.expires_on - token.issued_at) / 2)
        return token if _now() < token.expires_on - buffer else None

    def _valid(self) -> AccessToken | None:
        """The current token while Platform still accepts it, due for a refresh or not."""
        if self._current and _now() < self._current.expires_on:
            return self._current
        return None

    def _begin_mint(self) -> dict[str, Any]:
        """First half of a mint: the 429 gate, then the kwargs for the client-token POST."""
        if _now() < self._blocked_until:
            raise RateLimitedError(
                "client-token requests are rate limited; not minting until the cooldown ends"
            )
        kwargs: dict[str, Any] = {
            "json": {"client_id": self.client_id, "client_secret": self._client_secret}
        }
        merge_request_attribution(kwargs)
        return kwargs

    def _finish_mint(self, response: httpx.Response) -> AccessToken:
        """Second half of a mint: the response becomes the current token, or raises."""
        try:
            raise_for_status(response)
        except RateLimitedError:
            self._blocked_until = _now() + RATE_LIMIT_COOLDOWN_S
            raise
        try:
            parsed = TokenResponse(**response.json())
        except (ValueError, TypeError) as e:
            raise VisionaiSDKError(f"client-token response is not a token: {e}") from e
        now = _now()
        self._current = AccessToken(
            parsed.access_token, now + parsed.expires_in, parsed.token_type, now
        )
        logger.info(
            "platform token minted",
            extra={"client_id": self.client_id, "expires_in": parsed.expires_in},
        )
        return self._current

    def _after_failed_refresh(self, error: VisionaiSDKError) -> AccessToken:
        """A refresh failed: keep serving the current token while it is valid, else raise."""
        valid = self._valid()
        if valid is None:
            raise error
        logger.warning(
            "token refresh failed; using the current token until it expires",
            extra={
                "client_id": self.client_id,
                "error": str(error),
                "expires_in": valid.expires_in(),
            },
        )
        return valid

    def invalidate(self, token: str) -> bool:
        """Report a 401 for ``token``. True: dropped, so the next ``get_token()`` mints.

        A late report for a token no longer held changes nothing. A token minted
        less than ``REPLACE_COOLDOWN_S`` ago is kept: a 401 that soon is a client or
        allowlist problem another mint would not fix, and the cooldown bounds how
        often a persistent 401 can mint. While a mint is in flight the report is
        already being acted on.
        """
        if self._current is None or token != self._current.token:
            return False
        if self._minting:
            return True
        if _now() - self._current.issued_at < REPLACE_COOLDOWN_S:
            logger.warning(
                "401 on a token minted %d s ago; keeping it until the cooldown ends",
                int(_now() - self._current.issued_at),
                extra={"client_id": self.client_id},
            )
            return False
        self._current = None
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
        self._http_owner = http is None  # a client we built is ours to close
        self._lock = threading.Lock()

    def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        """The cached token, refreshed shortly before it expires.

        ``force_refresh=True`` mints even while the cached token is fresh. A refresh
        that fails keeps serving the current token while Platform still accepts it.

        Raises:
            AuthenticationError: Platform refused the client credentials (401).
            RateLimitedError: Platform answered 429, or did so less than an hour ago.
            NetworkError: Platform could not be reached.
            VisionaiSDKError: Any other failure.
        """
        with self._lock:
            if not force_refresh:
                fresh = self._fresh()
                if fresh is not None:
                    return fresh
            self._minting = True
            try:
                kwargs = self._begin_mint()
                try:
                    response = self._http.post(self._url, **kwargs)
                except httpx.RequestError as e:
                    raise from_transport_error(e) from e
                return self._finish_mint(response)
            except VisionaiSDKError as e:
                if force_refresh:
                    raise
                return self._after_failed_refresh(e)
            finally:
                self._minting = False

    def close(self) -> None:
        if self._http_owner:
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
        self._http_owner = http is None  # a client we built is ours to close
        self._lock = asyncio.Lock()

    async def get_token(self, *, force_refresh: bool = False) -> AccessToken:
        """The cached token, refreshed shortly before it expires.

        Concurrent callers wait for one mint. Behaves and raises as
        ``ClientCredentials.get_token``.
        """
        async with self._lock:
            if not force_refresh:
                fresh = self._fresh()
                if fresh is not None:
                    return fresh
            self._minting = True
            try:
                kwargs = self._begin_mint()
                try:
                    response = await self._http.post(self._url, **kwargs)
                except httpx.RequestError as e:
                    raise from_transport_error(e) from e
                return self._finish_mint(response)
            except VisionaiSDKError as e:
                if force_refresh:
                    raise
                return self._after_failed_refresh(e)
            finally:
                self._minting = False

    async def aclose(self) -> None:
        if self._http_owner:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClientCredentials:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
