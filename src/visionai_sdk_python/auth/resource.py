"""Authentication resource for sync operations."""

import logging
from typing import TYPE_CHECKING

import jwt

from ..endpoints import AuthEndpoint
from ..exceptions import JwksDiscoveryError
from .models import TokenResponse
from ._mixin import AuthMixin

if TYPE_CHECKING:
    from ..client import Client

logger = logging.getLogger(__name__)


class AuthResource(AuthMixin):
    """Synchronous authentication operations."""

    def __init__(self, client: "Client") -> None:
        """Initialize auth resource.

        Args:
            client: Parent sync client instance
        """
        self._client = client

    def get_access_token(self, client_id: str, client_secret: str) -> TokenResponse:
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
        # Validate (from Mixin)
        self._validate_client_credentials(client_id, client_secret)

        # I/O operation (sync)
        response = self._client._request(
            "POST",
            self._client._build_url(
                self._client.auth_url, AuthEndpoint.CLIENT_TOKEN
            ),
            json=self._prepare_client_token_request(client_id, client_secret),
        )

        # Parse and store (from Mixin)
        return self._parse_and_store_token(
            response.json(),
            credentials={"client_id": client_id, "client_secret": client_secret},
            credentials_type="client",
        )

    def login(self, email: str, password: str) -> TokenResponse:
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
        # Validate (from Mixin)
        self._validate_login_credentials(email, password)

        # I/O operation (sync)
        response = self._client._request(
            "POST",
            self._client._build_url(self._client.auth_url, AuthEndpoint.LOGIN),
            json=self._prepare_login_request(email, password),
        )

        # Parse and store (from Mixin)
        return self._parse_and_store_token(
            response.json(),
            credentials={"email": email, "password": password},
            credentials_type="login",
        )

    def is_token_valid(self, access_token: str) -> bool:
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
            # I/O operation (sync)
            self._client._jwt_verifier.verify_sync(access_token)
            return True
        except jwt.InvalidTokenError as e:
            self._log_token_validation_error(e, "InvalidTokenError")
            return False
        except jwt.PyJWKClientError as e:
            self._log_token_validation_error(e, "PyJWKClientError")
            return False
        except JwksDiscoveryError as e:
            self._log_token_validation_error(e, "JwksDiscoveryError")
            return False
