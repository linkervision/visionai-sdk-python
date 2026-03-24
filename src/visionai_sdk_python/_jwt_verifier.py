import asyncio
import time

import httpx
import jwt
from jwt import PyJWKClient

_OIDC_DISCOVERY_PATH = "/.well-known/openid-configuration"
_JWKS_URI_TTL: float = 3600.0  # seconds; refresh OIDC discovery cache after 1 hour
# Supported asymmetric signing algorithms:
# - RS256/384/512: RSA signature with SHA-256/384/512
# - ES256/384/512: ECDSA signature with SHA-256/384/512
_ALLOWED_ALGORITHMS: list[str] = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]


class JwtVerifier:
    """Stateful JWT verifier that handles OIDC discovery and JWKS key fetching.

    Maintains two internal caches:
    - ``_jwks_uri_cache``: issuer → jwks_uri with TTL-based expiry.
    - ``_jwks_clients``: jwks_uri → PyJWKClient (key-level caching delegated to PyJWKClient).
    """

    def __init__(self, verify_ssl: bool = True, timeout: float = 10.0) -> None:
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        # issuer -> (jwks_uri, expire_at)
        self._jwks_uri_cache: dict[str, tuple[str, float]] = {}
        # jwks_uri -> PyJWKClient
        self._jwks_clients: dict[str, PyJWKClient] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached_jwks_uri(self, issuer: str) -> str | None:
        entry = self._jwks_uri_cache.get(issuer)
        if entry is None:
            return None
        jwks_uri, expire_at = entry
        if time.monotonic() >= expire_at:
            del self._jwks_uri_cache[issuer]
            return None
        return jwks_uri

    def _cache_jwks_uri(self, issuer: str, jwks_uri: str) -> None:
        self._jwks_uri_cache[issuer] = (jwks_uri, time.monotonic() + _JWKS_URI_TTL)

    def _get_jwks_client(self, jwks_uri: str) -> PyJWKClient:
        if jwks_uri not in self._jwks_clients:
            self._jwks_clients[jwks_uri] = PyJWKClient(
                jwks_uri,
                cache_keys=True,
                timeout=int(self._timeout),
            )
        return self._jwks_clients[jwks_uri]

    # ------------------------------------------------------------------
    # Pure JWT helpers (no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_issuer(access_token: str) -> str:
        """Return issuer from the token without verifying signature."""
        claims = jwt.decode(
            access_token,
            options={"verify_signature": False, "verify_exp": False},
            algorithms=_ALLOWED_ALGORITHMS,
        )
        issuer: str = claims.get("iss", "")
        if not issuer:
            raise jwt.MissingRequiredClaimError("iss")
        return issuer

    @staticmethod
    def _decode_verified(access_token: str, signing_key: object) -> dict:
        """Verify signature + exp and return claims."""
        return jwt.decode(
            access_token,
            signing_key,  # type: ignore[arg-type]
            algorithms=_ALLOWED_ALGORITHMS,
            options={"verify_exp": True, "verify_aud": False},
        )

    # ------------------------------------------------------------------
    # OIDC discovery: sync / async
    # ------------------------------------------------------------------

    def _fetch_jwks_uri_sync(self, issuer: str) -> str:
        """Resolve jwks_uri from the OIDC discovery document (blocking)."""
        cached = self._get_cached_jwks_uri(issuer)
        if cached is not None:
            return cached

        discovery_url = f"{issuer.rstrip('/')}{_OIDC_DISCOVERY_PATH}"
        with httpx.Client(verify=self._verify_ssl, timeout=self._timeout) as http:
            resp = http.get(discovery_url)
            resp.raise_for_status()
            jwks_uri: str = resp.json()["jwks_uri"]

        self._cache_jwks_uri(issuer, jwks_uri)
        return jwks_uri

    async def _fetch_jwks_uri_async(self, issuer: str) -> str:
        """Resolve jwks_uri from the OIDC discovery document (non-blocking)."""
        cached = self._get_cached_jwks_uri(issuer)
        if cached is not None:
            return cached

        discovery_url = f"{issuer.rstrip('/')}{_OIDC_DISCOVERY_PATH}"
        async with httpx.AsyncClient(verify=self._verify_ssl, timeout=self._timeout) as http:
            resp = await http.get(discovery_url)
            resp.raise_for_status()
            jwks_uri: str = resp.json()["jwks_uri"]

        self._cache_jwks_uri(issuer, jwks_uri)
        return jwks_uri

    # ------------------------------------------------------------------
    # Public verify API
    # ------------------------------------------------------------------

    def verify_sync(self, access_token: str) -> dict:
        """Validate JWT signature and expiration (blocking).

        Raises:
            jwt.ExpiredSignatureError: Token has expired.
            jwt.InvalidSignatureError: Signature does not match.
            jwt.DecodeError: Token is malformed.
            jwt.MissingRequiredClaimError: Token is missing the ``iss`` claim.
        """
        issuer = self._get_issuer(access_token)
        jwks_uri = self._fetch_jwks_uri_sync(issuer)
        signing_key = self._get_jwks_client(jwks_uri).get_signing_key_from_jwt(access_token)
        return self._decode_verified(access_token, signing_key.key)

    async def verify_async(self, access_token: str) -> dict:
        """Validate JWT signature and expiration (non-blocking).

        Raises:
            jwt.ExpiredSignatureError: Token has expired.
            jwt.InvalidSignatureError: Signature does not match.
            jwt.DecodeError: Token is malformed.
            jwt.MissingRequiredClaimError: Token is missing the ``iss`` claim.
        """
        issuer = self._get_issuer(access_token)
        jwks_uri = await self._fetch_jwks_uri_async(issuer)
        # PyJWKClient.get_signing_key_from_jwt does blocking HTTP; offload to thread pool
        signing_key = await asyncio.to_thread(
            self._get_jwks_client(jwks_uri).get_signing_key_from_jwt, access_token
        )
        return self._decode_verified(access_token, signing_key.key)
