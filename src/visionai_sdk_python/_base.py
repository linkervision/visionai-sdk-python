import time
from typing import TYPE_CHECKING, Literal

import httpx

from ._jwt_verifier import JwtVerifier
from .constants import resolve_allowed_issuers
from .exceptions import raise_for_status

if TYPE_CHECKING:
    from .auth.credentials import AsyncClientCredentials, ClientCredentials


class _BaseClient:
    """Base class for VisionAI SDK clients.

    Holds shared connection configuration and provides
    common URL/header builder utilities.
    """

    def __init__(
        self,
        auth_url: str,
        vlm_url: str | None = None,
        allowed_issuers: list[str] | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        credential: "ClientCredentials | AsyncClientCredentials | None" = None,
    ) -> None:
        """Initialize the client with connection settings.

        Args:
            auth_url: Base URL for the authentication service.
            vlm_url: Base URL for the VLM inference service. Optional for clients
                that only authenticate; VLM calls then raise ``ValueError``.
            allowed_issuers: Optional list of allowed JWT issuers. If provided, tokens
                whose ``iss`` claim is not in this list will be rejected. If omitted,
                the issuers are derived from ``auth_url``.
            verify_ssl: Whether to verify TLS certificates.
            timeout: Default request timeout in seconds.
            max_connections: Maximum number of concurrent connections in the pool.
            max_keepalive_connections: Maximum number of idle keep-alive connections.
            credential: A ``ClientCredentials`` / ``AsyncClientCredentials`` whose
                token every call uses; shareable between clients. Without it the
                client keeps the token from ``auth.login()`` or ``set_token()``;
                ``auth.get_access_token()`` installs a credential. The last explicit
                authentication wins.
        """
        if not auth_url.strip():
            raise ValueError("auth_url must not be empty")
        if vlm_url is not None and not vlm_url.strip():
            raise ValueError("vlm_url must not be empty")
        self.auth_url = auth_url.strip()
        self.vlm_url = vlm_url.strip() if vlm_url is not None else None
        self._credential = credential
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        resolved_issuers: list[str] = (
            allowed_issuers
            if allowed_issuers is not None
            else resolve_allowed_issuers(self.auth_url)
        )
        self._jwt_verifier = JwtVerifier(
            auth_url=self.auth_url,
            allowed_issuers=resolved_issuers,
            verify_ssl=verify_ssl,
            timeout=timeout,
        )

        # Token management state
        self._access_token: str | None = None
        self._token_expires_at: float | None = None  # monotonic time
        self._credentials: dict | None = None
        self._credentials_type: Literal["login"] | None = None

    @staticmethod
    def _build_url(base_url: str, path: str) -> str:
        """Join a base URL and a path, normalizing slashes."""
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    def _require_vlm_url(self) -> str:
        if self.vlm_url is None:
            raise ValueError("vlm_url is required for VLM calls; pass it to the client")
        return self.vlm_url

    @staticmethod
    def _build_auth_header(access_token: str) -> dict[str, str]:
        """Build authorization header."""
        if not access_token:
            raise ValueError("access_token must not be empty")
        return {"Authorization": f"Bearer {access_token}"}

    def _store_token(
        self,
        access_token: str,
        expires_in: int,
        credentials: dict | None = None,
        credentials_type: Literal["login"] | None = None,
    ) -> None:
        """Use this token from now on; a credential set earlier is dropped.

        The last explicit authentication wins, in both directions.

        Args:
            access_token: JWT access token.
            expires_in: Token expiration time in seconds.
            credentials: Optional credentials for auto-refresh.
            credentials_type: "login" for email / password credentials; client
                credentials live in a ``ClientCredentials`` on the client instead.
        """
        self._credential = None
        self._access_token = access_token
        self._token_expires_at = time.monotonic() + expires_in
        self._credentials = credentials
        self._credentials_type = credentials_type

    def _is_token_expiring_soon(self, buffer_seconds: int = 30) -> bool:
        """Check if token is expiring soon.

        Args:
            buffer_seconds: Time buffer in seconds before actual expiration.

        Returns:
            True if token is expiring within buffer_seconds, False otherwise.
        """
        if self._token_expires_at is None:
            return True
        return time.monotonic() >= (self._token_expires_at - buffer_seconds)

    def set_token(self, access_token: str, expires_in: int | None = None) -> None:
        """Set externally obtained token.

        Use this when you have an access token obtained through other means
        (e.g., frontend authorization code flow). Note that tokens set this way
        cannot be auto-refreshed since no credentials are stored.

        The token will be validated locally to ensure it has a valid signature
        and has not expired. The actual expiration time from the token's ``exp``
        claim will be used. If ``expires_in`` is provided, the minimum of the two
        will be used to prevent extending the token's lifetime beyond its true expiration.

        Args:
            access_token: JWT access token.
            expires_in: Optional token expiration time in seconds. If provided,
                the effective expiration will be min(expires_in, token's actual remaining time).
                If None, the token's actual ``exp`` claim will be used.

        Raises:
            ValueError: If access_token is empty.
            jwt.ExpiredSignatureError: If token has already expired.
            jwt.InvalidSignatureError: If token signature is invalid.
            jwt.DecodeError: If token is malformed.
            jwt.MissingRequiredClaimError: If token is missing required claims.
            jwt.InvalidIssuerError: If token issuer is not allowed.

        Example:
            >>> client = Client(auth_url="...", vlm_url="...")
            >>> client.set_token("eyJhbG...")
            >>> result = client.chat(payload)  # Token will be used automatically
        """
        if not access_token.strip():
            raise ValueError("access_token must not be empty")

        # Validate token and extract exp claim
        claims = self._jwt_verifier.verify_sync(access_token)
        jwt_exp = claims.get("exp")
        if jwt_exp is None:
            raise ValueError("Token missing 'exp' claim")

        # Calculate remaining time from JWT exp
        jwt_expires_in = int(jwt_exp - time.time())
        if jwt_expires_in <= 0:
            raise ValueError("Token has already expired")

        # Use the minimum of provided expires_in and actual JWT expiration
        if expires_in is not None:
            effective_expires_in = min(expires_in, jwt_expires_in)
        else:
            effective_expires_in = jwt_expires_in

        self._store_token(
            access_token, effective_expires_in, credentials=None, credentials_type=None
        )

    @staticmethod
    def _handle_response(response: httpx.Response) -> httpx.Response:
        """Raise the SDK exception for a non-2xx response (``exceptions.raise_for_status``)."""
        return raise_for_status(response)
