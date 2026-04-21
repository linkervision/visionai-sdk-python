"""Shared authentication business logic for sync and async resources."""

import logging

import jwt

from ..exceptions import JwksDiscoveryError
from .models import TokenResponse

logger = logging.getLogger(__name__)


class AuthMixin:
    """Shared authentication logic (validation, data preparation, parsing).

    This mixin contains all business logic that doesn't involve I/O operations.
    Sync and async resources inherit from this to avoid code duplication.
    """

    # ==========================================================================
    # Validation methods
    # ==========================================================================

    def _validate_login_credentials(self, email: str, password: str) -> None:
        """Validate login credentials.

        Args:
            email: User email address
            password: User password

        Raises:
            ValueError: If email or password is empty
        """
        if not email.strip():
            raise ValueError("email must not be empty")
        if not password.strip():
            raise ValueError("password must not be empty")

    def _validate_client_credentials(
        self, client_id: str, client_secret: str
    ) -> None:
        """Validate client credentials.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret

        Raises:
            ValueError: If client_id or client_secret is empty
        """
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if not client_secret.strip():
            raise ValueError("client_secret must not be empty")

    # ==========================================================================
    # Request preparation methods
    # ==========================================================================

    def _prepare_login_request(self, email: str, password: str) -> dict:
        """Prepare login request payload.

        Args:
            email: User email address
            password: User password

        Returns:
            Request payload dict
        """
        return {"email": email, "password": password}

    def _prepare_client_token_request(
        self, client_id: str, client_secret: str
    ) -> dict:
        """Prepare client token request payload.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret

        Returns:
            Request payload dict
        """
        return {"client_id": client_id, "client_secret": client_secret}

    # ==========================================================================
    # Response parsing and token storage
    # ==========================================================================

    def _parse_and_store_token(
        self, response_data: dict, credentials: dict, credentials_type: str
    ) -> TokenResponse:
        """Parse token response and store it in the client.

        Args:
            response_data: Response JSON data
            credentials: Credentials dict for auto-refresh
            credentials_type: Type of credentials ("login" or "client")

        Returns:
            Parsed TokenResponse
        """
        token = TokenResponse(**response_data)
        self._client._store_token(
            access_token=token.access_token,
            expires_in=token.expires_in,
            credentials=credentials,
            credentials_type=credentials_type,
        )
        return token

    # ==========================================================================
    # Token validation exception handling
    # ==========================================================================

    def _handle_token_validation_exceptions(self, access_token: str) -> bool:
        """Handle exceptions during token validation.

        This method encapsulates the exception handling logic for is_token_valid().
        Subclasses should call this with the appropriate verify method.

        Args:
            access_token: JWT access token to validate

        Returns:
            True if validation succeeds, False if expected exceptions occur

        Note:
            This method should be called within a try-except block that calls
            the appropriate verify method (verify_sync or verify_async).
        """
        # This method will be called from subclasses after they attempt verification
        # The actual verification is in the try block of the caller
        pass

    def _log_token_validation_error(self, error: Exception, error_type: str) -> None:
        """Log token validation errors.

        Args:
            error: The exception that occurred
            error_type: Type of error for logging categorization
        """
        if isinstance(error, jwt.InvalidTokenError):
            logger.warning(
                "%s: Token validation failed",
                type(error).__name__,
                extra={
                    "jwt_error_type": type(error).__name__,
                    "jwt_error_message": str(error),
                },
            )
        elif isinstance(error, jwt.PyJWKClientError):
            logger.error(
                "%s: JWKS client error during token validation",
                type(error).__name__,
                extra={
                    "jwt_error_type": "PyJWKClientError",
                    "jwt_error_message": str(error),
                },
            )
        elif isinstance(error, JwksDiscoveryError):
            logger.error(
                "%s: OIDC discovery failed during token validation",
                type(error).__name__,
                extra={
                    "jwt_error_type": type(error).__name__,
                    "jwt_error_message": str(error),
                },
            )
