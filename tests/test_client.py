import pytest
import httpx
import jwt
from unittest.mock import patch

from pydantic import ValidationError

from visionai_sdk_python.client import Client
from visionai_sdk_python.exceptions import APIError, AuthenticationError, JwksDiscoveryError, NetworkError, ServerError
from visionai_sdk_python.auth.models import TokenResponse
from visionai_sdk_python.vlm.models import NIMRequestModel, ResponseErrorModel, ResponseNormalModel
from tests.constants import (
    AUTH_URL, VLM_URL, TOKEN_PAYLOAD,
    VALID_NIM_PAYLOAD,
    NORMAL_PENDING_RESPONSE, NORMAL_RUNNING_RESPONSE, NORMAL_COMPLETED_RESPONSE,
    ERROR_FAILED_RESPONSE, ERROR_TIMEOUT_RESPONSE,
)


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
    result = mock_client.auth.login("user@example.com", "correct-password")

    # Assert
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]
    assert result.expires_in == TOKEN_PAYLOAD["expires_in"]
    assert result.token_type == TOKEN_PAYLOAD["token_type"]
    # Assert store token
    assert mock_client._access_token == TOKEN_PAYLOAD["access_token"]
    assert mock_client._token_expires_at is not None
    assert mock_client._credentials == {"email": "user@example.com", "password": "correct-password"}
    assert mock_client._credentials_type == "login"

 
def test_login_wrong_credentials(unauthorized_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=unauthorized_transport)

    # Act & Assert
    with pytest.raises(AuthenticationError, match="Invalid credentials") as exc_info:
        c.auth.login("user@example.com", "wrong-password")
    assert exc_info.value.status_code == 401


def test_login_server_error(server_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=server_error_transport)

    # Act & Assert
    with pytest.raises(ServerError, match="Service Unavailable") as exc_info:
        c.auth.login("user@example.com", "password")
    assert exc_info.value.status_code == 503


def test_login_network_error(connect_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=connect_error_transport)

    # Act & Assert
    with pytest.raises(NetworkError, match="Network error"):
        c.auth.login("user@example.com", "password")


def test_login_api_error(api_error_transport):
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=api_error_transport)

    # Act & Assert
    with pytest.raises(APIError, match="Too many redirect") as exc_info:
        c.auth.login("user@example.com", "correct-password")
    assert exc_info.value.status_code == 310


def test_login_empty_email(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="email must not be empty"):
        mock_client.auth.login("", "password")


def test_login_empty_password(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="password must not be empty"):
        mock_client.auth.login("user@example.com", "")


def test_login_whitespace_email(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="email must not be empty"):
        mock_client.auth.login("   ", "password")


def test_login_whitespace_password(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="password must not be empty"):
        mock_client.auth.login("user@example.com", "   ")


# --- get_access_token ---

def test_get_access_token_success(mock_client: Client) -> None:
    # Arrange: mock_client fixture provides a client with successful transport

    # Act
    result = mock_client.auth.get_access_token("admin", "secret")

    # Assert
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]
    # Assert store token
    assert mock_client._access_token == TOKEN_PAYLOAD["access_token"]
    assert mock_client._is_token_expiring_soon is not None
    assert mock_client._credentials == {"client_id": "admin", "client_secret": "secret"}
    assert mock_client._credentials_type == 'client'


def test_get_access_token_wrong_credentials(unauthorized_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=unauthorized_transport)

    # Act & Assert
    with pytest.raises(AuthenticationError, match="Invalid credentials") as exc_info:
        c.auth.get_access_token("wrong-id", "wrong-secret")
    assert exc_info.value.status_code == 401


def test_get_access_token_server_error(server_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=server_error_transport)

    # Act & Assert
    with pytest.raises(ServerError, match="Service Unavailable") as exc_info:
        c.auth.get_access_token("admin", "secret")
    assert exc_info.value.status_code == 503


def test_get_access_token_network_error(connect_error_transport: httpx.MockTransport) -> None:
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=connect_error_transport)

    # Act & Assert
    with pytest.raises(NetworkError, match="Network error") as exc_info:
        c.auth.get_access_token("admin", "secret")


def test_get_access_token_api_error(api_error_transport):
    # Arrange
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=api_error_transport)

    # Act & Assert
    with pytest.raises(APIError, match="Too many redirect") as exc_info:
        c.auth.get_access_token("admin", "secret")
    assert exc_info.value.status_code == 310


def test_get_access_token_empty_client_id(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_id must not be empty"):
        mock_client.auth.get_access_token("", "secret")


def test_get_access_token_empty_client_secret(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        mock_client.auth.get_access_token("client-id", "")


def test_get_access_token_whitespace_client_id(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_id must not be empty"):
        mock_client.auth.get_access_token("   ", "secret")


def test_get_access_token_whitespace_client_secret(mock_client: Client) -> None:
    # Arrange: mock_client fixture

    # Act & Assert
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        mock_client.auth.get_access_token("client-id", "   ")


# --- is_token_valid ---

def test_is_token_valid_returns_true_on_valid_token(mock_client: Client) -> None:
    # Arrange
    claims = {"sub": "user-123", "iss": "https://auth.example.com"}
    with patch.object(mock_client._jwt_verifier, "verify_sync", return_value=claims):
        # Act
        result = mock_client.auth.is_token_valid("any.jwt.token")

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "exc,exc_name",
    [
        (jwt.ExpiredSignatureError("Token has expired"), "ExpiredSignatureError"),
        (jwt.InvalidSignatureError("Signature verification failed"), "InvalidSignatureError"),
        (jwt.DecodeError("Not enough segments"), "DecodeError"),
        (jwt.MissingRequiredClaimError("iss"), "MissingRequiredClaimError"),
        (jwt.InvalidIssuerError("Token issuer 'https://evil.com/' is not in the allowed issuers list"), "InvalidIssuerError"),
    ],
    ids=["expired", "invalid_signature", "malformed", "missing_iss", "invalid_issuer"],
)
def test_is_token_valid_logs_warning_on_invalid_token(
    mock_client: Client, exc: jwt.InvalidTokenError, exc_name: str, caplog
) -> None:
    # Arrange
    from visionai_sdk_python.auth import resource as auth_resource
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act
        with caplog.at_level("WARNING", logger=auth_resource.__name__):
            result = mock_client.auth.is_token_valid("any.jwt.token")

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
    from visionai_sdk_python.auth import resource as auth_resource
    exc = jwt.PyJWKClientError("Failed to fetch JWKS from endpoint")
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act
        with caplog.at_level("ERROR", logger=auth_resource.__name__):
            result = mock_client.auth.is_token_valid("any.jwt.token")

    # Assert
    assert result is False
    assert len(caplog.records) == 1

    log_record = caplog.records[0]
    assert log_record.levelname == "ERROR"
    assert "JWKS client error during token validation" in log_record.message
    assert log_record.jwt_error_type == "PyJWKClientError"
    assert log_record.jwt_error_message == str(exc)


def test_is_token_valid_logs_error_on_jwks_discovery_failure(
    mock_client: Client, caplog
) -> None:
    # Arrange
    from visionai_sdk_python.auth import resource as auth_resource
    exc = JwksDiscoveryError("Failed to fetch OIDC discovery document from 'https://auth.example.com/.well-known/openid-configuration': [Errno -2] Name or service not known")
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act
        with caplog.at_level("ERROR", logger=auth_resource.__name__):
            result = mock_client.auth.is_token_valid("any.jwt.token")

    # Assert
    assert result is False
    assert len(caplog.records) == 1

    log_record = caplog.records[0]
    assert "OIDC discovery failed during token validation" in log_record.message
    assert log_record.jwt_error_type == "JwksDiscoveryError"
    assert log_record.jwt_error_message == str(exc)


def test_is_token_valid_propagates_unexpected_errors(
    mock_client: Client
) -> None:
    # Arrange - Simulate a programming error (e.g., AttributeError)
    exc = AttributeError("'NoneType' object has no attribute 'verify'")
    with patch.object(mock_client._jwt_verifier, "verify_sync", side_effect=exc):
        # Act & Assert - Unexpected exceptions should propagate
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'verify'"):
            mock_client.auth.is_token_valid("any.jwt.token")


# --- Resource cleanup ---

def test_client_context_manager(success_transport: httpx.MockTransport) -> None:
    """Test that client properly closes resources when used as context manager."""
    # Arrange
    with Client(auth_url=AUTH_URL, vlm_url=VLM_URL) as client:
        client._client = httpx.Client(transport=success_transport)

        # Act
        result = client.auth.login("user@example.com", "password")

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
    result = c.auth.login("user@example.com", "password")

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


# --- chat / get_chat shared fixtures & helpers ---

def _make_vlm_client(response_body: dict) -> Client:
    """Return a Client whose transport always returns response_body and that is authenticated."""
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response_body))
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=transport)
    # Simulate authenticated state with a valid token
    c._store_token(
        access_token="valid.jwt.token",
        expires_in=3600,
        credentials={"email": "test@example.com", "password": "password"},
        credentials_type="login",
    )
    return c


# --- chat ---

def test_chat_raises_authentication_error_when_not_authenticated(mock_client: Client) -> None:
    # Arrange: client has no token
    nim = NIMRequestModel(**VALID_NIM_PAYLOAD)

    # Act & Assert: chat() raises AuthenticationError
    with pytest.raises(AuthenticationError, match="Not authenticated"):
        mock_client.vlm.chat(nim)


def test_chat_raises_authentication_error_when_token_expired_without_credentials(mock_client: Client) -> None:
    # Arrange: client has token but no credentials, and token is expiring
    mock_client._access_token = "fake_token"
    mock_client._credentials = None

    # Act & Assert: chat() raises AuthenticationError when token expires
    with patch.object(mock_client, "_is_token_expiring_soon", return_value=True):
        nim = NIMRequestModel(**VALID_NIM_PAYLOAD)
        with pytest.raises(AuthenticationError, match="no credentials available"):
            mock_client.vlm.chat(nim)


def test_chat_auto_refreshes_token_when_expiring_soon() -> None:
    # Arrange: client with expiring token and valid credentials
    c = _make_vlm_client(NORMAL_COMPLETED_RESPONSE)
    nim = NIMRequestModel(**VALID_NIM_PAYLOAD)

    # Act: call chat() when token is expiring
    with patch.object(c, "_is_token_expiring_soon", return_value=True):
        with patch.object(c.auth, "login", return_value=TokenResponse(**TOKEN_PAYLOAD)) as mock_login:
            c.vlm.chat(nim)

            # Assert: login was called to refresh token
            mock_login.assert_called_once_with(
                email="test@example.com",
                password="password"
            )


def test_chat_invalid_dict_payload_raises_validation_error(mock_client: Client) -> None:
    # Arrange: client is authenticated, payload missing required fields
    mock_client._store_token("valid.jwt.token", 3600, {"email": "test@example.com", "password": "password"}, "login")
    # Act & Assert
    with pytest.raises(ValidationError):
        mock_client.vlm.chat({"bad_field": "value"})


def test_chat_with_valid_nim_model_returns_normal_model() -> None:
    # Arrange
    c = _make_vlm_client(NORMAL_COMPLETED_RESPONSE)
    nim = NIMRequestModel(**VALID_NIM_PAYLOAD)

    # Act
    result = c.vlm.chat(nim)

    # Assert
    assert isinstance(result, ResponseNormalModel)
    assert result.status == "completed"
    assert result.chat_id == "id-001"


@pytest.mark.parametrize("response_body,expected_status", [
    (NORMAL_PENDING_RESPONSE,   "pending"),
    (NORMAL_RUNNING_RESPONSE,   "running"),
    (NORMAL_COMPLETED_RESPONSE, "completed"),
], ids=["pending", "running", "completed"])
def test_chat_returns_normal_model_for_non_error_statuses(
    response_body: dict, expected_status: str
) -> None:
    # Arrange
    c = _make_vlm_client(response_body)

    # Act
    result = c.vlm.chat(VALID_NIM_PAYLOAD)

    # Assert
    assert isinstance(result, ResponseNormalModel)
    assert result.status == expected_status


@pytest.mark.parametrize("response_body,expected_status", [
    (ERROR_FAILED_RESPONSE,  "failed"),
    (ERROR_TIMEOUT_RESPONSE, "timeout"),
], ids=["failed", "timeout"])
def test_chat_returns_error_model_for_error_statuses(
    response_body: dict, expected_status: str
) -> None:
    # Arrange
    c = _make_vlm_client(response_body)

    # Act
    result = c.vlm.chat(VALID_NIM_PAYLOAD)

    # Assert
    assert isinstance(result, ResponseErrorModel)
    assert result.status == expected_status


# --- get_chat ---

@pytest.mark.parametrize("response_body,expected_status", [
    (NORMAL_PENDING_RESPONSE,   "pending"),
    (NORMAL_RUNNING_RESPONSE,   "running"),
    (NORMAL_COMPLETED_RESPONSE, "completed"),
], ids=["pending", "running", "completed"])
def test_get_chat_returns_normal_model_for_non_error_statuses(
    response_body: dict, expected_status: str
) -> None:
    # Arrange
    c = _make_vlm_client(response_body)

    # Act
    result = c.vlm.get_chat("id-001")

    # Assert
    assert isinstance(result, ResponseNormalModel)
    assert result.status == expected_status
    assert result.chat_id == "id-001"


@pytest.mark.parametrize("response_body,expected_status", [
    (ERROR_FAILED_RESPONSE,  "failed"),
    (ERROR_TIMEOUT_RESPONSE, "timeout"),
], ids=["failed", "timeout"])
def test_get_chat_returns_error_model_for_error_statuses(
    response_body: dict, expected_status: str
) -> None:
    # Arrange
    c = _make_vlm_client(response_body)

    # Act
    result = c.vlm.get_chat("id-001")

    # Assert
    assert isinstance(result, ResponseErrorModel)
    assert result.status == expected_status
    assert result.error is not None


