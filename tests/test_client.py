import pytest
import httpx
import jwt
from unittest.mock import patch

from visionai_sdk_python.client import Client
from visionai_sdk_python.exceptions import APIError, AuthenticationError, NetworkError, ServerError
from visionai_sdk_python.models import TokenResponse

AUTH_URL = "https://auth.example.com"
VLM_URL = "https://vlm.example.com"

TOKEN_PAYLOAD = {
    "access_token": "test-jwt-token",
    "expires_in": 3600,
    "token_type": "Bearer",
}


@pytest.fixture
def success_transport() -> httpx.MockTransport:
    """Transport that returns successful token response."""
    return httpx.MockTransport(lambda r: httpx.Response(200, json=TOKEN_PAYLOAD))


@pytest.fixture
def unauthorized_transport() -> httpx.MockTransport:
    """Transport that returns 401 Unauthorized."""
    return httpx.MockTransport(
        lambda r: httpx.Response(401, json={"detail": "Invalid credentials"})
    )


@pytest.fixture
def server_error_transport() -> httpx.MockTransport:
    """Transport that returns 503 Service Unavailable."""
    return httpx.MockTransport(
        lambda r: httpx.Response(503, json={"detail": "Service Unavailable"})
    )


@pytest.fixture
def connect_error_transport() -> httpx.MockTransport:
    """Transport that raises ConnectError."""
    def _raise(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")
    return httpx.MockTransport(_raise)


@pytest.fixture
def api_error_transport() -> httpx.MockTransport:
    """Transport that returns 310 Too many redirect."""
    return httpx.MockTransport(
        lambda r: httpx.Response(310, json={"detail": "Too many redirect"})
    )


@pytest.fixture
def mock_client(success_transport: httpx.MockTransport) -> Client:
    """Create a Client with mocked successful transport."""
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=success_transport)
    return c


# --- login ---

def test_login_success(mock_client: Client) -> None:
    # Arrange: mock_client fixture provides a client with successful transport

    # Act
    result = mock_client.login("user@example.com", "correct-password")

    # Assert
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]
    assert result.expires_in == TOKEN_PAYLOAD["expires_in"]
    assert result.token_type == TOKEN_PAYLOAD["token_type"]


def test_login_wrong_credentials(unauthorized_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=unauthorized_transport)

    # Act & Assert
    with pytest.raises(AuthenticationError, match="Invalid credentials") as exc_info:
        c.login("user@example.com", "wrong-password")
    assert exc_info.value.status_code == 401


def test_login_server_error(server_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=server_error_transport)

    # Act & Assert
    with pytest.raises(ServerError, match="Service Unavailable") as exc_info:
        c.login("user@example.com", "password")
    assert exc_info.value.status_code == 503


def test_login_network_error(connect_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=connect_error_transport)

    # Act & Assert
    with pytest.raises(NetworkError, match="Network error"):
        c.login("user@example.com", "password")


@pytest.mark.asyncio
async def test_login_api_error(api_error_transport):
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=api_error_transport)

    # Act & Assert
    with pytest.raises(APIError, match="Too many redirect") as exc_info:
        c.login("user@example.com", "correct-password")
    assert exc_info.value.status_code == 310


def test_login_empty_email(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="email must not be empty"):
        mock_client.login("", "password")


def test_login_empty_password(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="password must not be empty"):
        mock_client.login("user@example.com", "")


def test_login_whitespace_email(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="email must not be empty"):
        mock_client.login("   ", "password")


def test_login_whitespace_password(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="password must not be empty"):
        mock_client.login("user@example.com", "   ")


# --- get_access_token ---

def test_get_access_token_success(mock_client: Client) -> None:
    # Arrange: mock_client fixture provides a client with successful transport

    # Act
    result = mock_client.get_access_token("admin", "secret")

    # Assert
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]


def test_get_access_token_wrong_credentials(unauthorized_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=unauthorized_transport)

    # Act & Assert
    with pytest.raises(AuthenticationError, match="Invalid credentials") as exc_info:
        c.get_access_token("wrong-id", "wrong-secret")
    assert exc_info.value.status_code == 401


def test_get_access_token_server_error(server_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=server_error_transport)

    # Act & Assert
    with pytest.raises(ServerError, match="Service Unavailable") as exc_info:
        c.get_access_token("admin", "secret")
    assert exc_info.value.status_code == 503


def test_get_access_token_network_error(connect_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=connect_error_transport)

    # Act & Assert
    with pytest.raises(NetworkError, match="Network error") as exc_info:
        c.get_access_token("admin", "secret")


@pytest.mark.asyncio
async def test_get_access_token_api_error(api_error_transport):
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=api_error_transport)

    # Act & Assert
    with pytest.raises(APIError, match="Too many redirect") as exc_info:
        c.get_access_token("admin", "secret")
    assert exc_info.value.status_code == 310


def test_get_access_token_empty_client_id(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_id must not be empty"):
        mock_client.get_access_token("", "secret")


def test_get_access_token_empty_client_secret(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        mock_client.get_access_token("client-id", "")


def test_get_access_token_whitespace_client_id(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_id must not be empty"):
        mock_client.get_access_token("   ", "secret")


def test_get_access_token_whitespace_client_secret(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        mock_client.get_access_token("client-id", "   ")


# --- is_token_valid ---

def test_is_token_valid_returns_true_on_valid_token(mock_client: Client) -> None:
    # Arrange
    claims = {"sub": "user-123", "iss": "https://auth.example.com"}
    with patch.object(mock_client._jwt_verifier, "verify_sync", return_value=claims):
        # Act
        result = mock_client.is_token_valid("any.jwt.token")

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "exc,exc_name",
    [
        (jwt.ExpiredSignatureError("Token has expired"), "ExpiredSignatureError"),
        (jwt.InvalidSignatureError("Signature verification failed"), "InvalidSignatureError"),
        (jwt.DecodeError("Not enough segments"), "DecodeError"),
        (jwt.MissingRequiredClaimError("iss"), "MissingRequiredClaimError"),
    ],
    ids=["expired", "invalid_signature", "malformed", "missing_iss"],
)
def test_is_token_valid_logs_warning_on_invalid_token(
    mock_client: Client, exc: jwt.InvalidTokenError, exc_name: str, caplog
) -> None:
    # Arrange
    from visionai_sdk_python import client
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act
        with caplog.at_level("WARNING", logger=client.__name__):
            result = mock_client.is_token_valid("any.jwt.token")

    # Assert
    assert result is False
    assert len(caplog.records) == 1

    log_record = caplog.records[0]
    assert log_record.levelname == "WARNING"
    assert "Token validation failed" in log_record.message
    assert exc_name in log_record.message
    assert log_record.jwt_error_type == exc_name
    assert log_record.jwt_error_message == str(exc)


def test_is_token_valid_logs_error_on_jwks_failure(
    mock_client: Client, caplog
) -> None:
    # Arrange
    from visionai_sdk_python import client
    exc = jwt.PyJWKClientError("Failed to fetch JWKS from endpoint")
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act
        with caplog.at_level("ERROR", logger=client.__name__):
            result = mock_client.is_token_valid("any.jwt.token")

    # Assert
    assert result is False
    assert len(caplog.records) == 1

    log_record = caplog.records[0]
    assert log_record.levelname == "ERROR"
    assert "JWKS client error during token validation" in log_record.message
    assert log_record.jwt_error_type == "PyJWKClientError"
    assert log_record.jwt_error_message == str(exc)


def test_is_token_valid_propagates_unexpected_errors(
    mock_client: Client
) -> None:
    # Arrange - Simulate a programming error (e.g., AttributeError)
    exc = AttributeError("'NoneType' object has no attribute 'verify'")
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act & Assert - Unexpected exceptions should propagate
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'verify'"):
            mock_client.is_token_valid("any.jwt.token")


# --- Resource cleanup ---

def test_client_context_manager(success_transport: httpx.MockTransport) -> None:
    """Test that client properly closes resources when used as context manager."""
    # Arrange
    with Client(auth_url=AUTH_URL, vlm_url=VLM_URL) as client:
        client._client = httpx.Client(transport=success_transport)

        # Act
        result = client.login("user@example.com", "password")

        # Assert
        assert isinstance(result, TokenResponse)

    # Assert: after exiting context manager, client should be closed
    assert client._client.is_closed


def test_client_explicit_close(success_transport: httpx.MockTransport) -> None:
    """Test that client can be explicitly closed."""
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=success_transport)

    # Act
    result = c.login("user@example.com", "password")

    # Assert: client is open after use
    assert isinstance(result, TokenResponse)
    assert not c._client.is_closed

    # Act: explicit close
    c.close()

    # Assert: client is closed
    assert c._client.is_closed


def test_client_close_idempotent(success_transport: httpx.MockTransport) -> None:
    """Test that calling close() multiple times is safe."""
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=success_transport)

    # Act
    c.close()
    c.close()  # Should not raise

    # Assert
    assert c._client.is_closed
