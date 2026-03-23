import pytest
import httpx

from visionai_sdk_python.client import Client
from visionai_sdk_python.exceptions import AuthenticationError, NetworkError, ServerError
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
def mock_client(success_transport: httpx.MockTransport) -> Client:
    """Create a Client with mocked successful transport."""
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=success_transport)
    return c


# --- login ---

def test_login_success(mock_client: Client) -> None:
    result = mock_client.login("user@example.com", "correct-password")
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]
    assert result.expires_in == TOKEN_PAYLOAD["expires_in"]
    assert result.token_type == TOKEN_PAYLOAD["token_type"]


def test_login_wrong_credentials(unauthorized_transport: httpx.MockTransport) -> None:
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=unauthorized_transport)

    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        c.login("user@example.com", "wrong-password")


def test_login_server_error(server_error_transport: httpx.MockTransport) -> None:
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=server_error_transport)

    with pytest.raises(ServerError) as exc_info:
        c.login("user@example.com", "password")
    assert exc_info.value.status_code == 503


def test_login_network_error(connect_error_transport: httpx.MockTransport) -> None:
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=connect_error_transport)

    with pytest.raises(NetworkError, match="Connection failed"):
        c.login("user@example.com", "password")


def test_login_empty_email(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="email must not be empty"):
        mock_client.login("", "password")


def test_login_empty_password(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="password must not be empty"):
        mock_client.login("user@example.com", "")


def test_login_whitespace_email(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="email must not be empty"):
        mock_client.login("   ", "password")


def test_login_whitespace_password(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="password must not be empty"):
        mock_client.login("user@example.com", "   ")


# --- get_access_token ---

def test_get_access_token_success(mock_client: Client) -> None:
    result = mock_client.get_access_token("platform-admin", "platform-admin-secret")
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]


def test_get_access_token_wrong_credentials(unauthorized_transport: httpx.MockTransport) -> None:
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=unauthorized_transport)

    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        c.get_access_token("wrong-id", "wrong-secret")


def test_get_access_token_server_error(server_error_transport: httpx.MockTransport) -> None:
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=server_error_transport)

    with pytest.raises(ServerError) as exc_info:
        c.get_access_token("platform-admin", "platform-admin-secret")
    assert exc_info.value.status_code == 503


def test_get_access_token_network_error(connect_error_transport: httpx.MockTransport) -> None:
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=connect_error_transport)

    with pytest.raises(NetworkError, match="Connection failed"):
        c.get_access_token("platform-admin", "platform-admin-secret")


def test_get_access_token_empty_client_id(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="client_id must not be empty"):
        mock_client.get_access_token("", "secret")


def test_get_access_token_empty_client_secret(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        mock_client.get_access_token("client-id", "")


def test_get_access_token_whitespace_client_id(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="client_id must not be empty"):
        mock_client.get_access_token("   ", "secret")


def test_get_access_token_whitespace_client_secret(mock_client: Client) -> None:
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        mock_client.get_access_token("client-id", "   ")


# --- Resource cleanup ---

def test_client_context_manager(success_transport: httpx.MockTransport) -> None:
    """Test that client properly closes resources when used as context manager."""
    with Client(auth_url=AUTH_URL, vlm_url=VLM_URL) as client:
        client._client = httpx.Client(transport=success_transport)
        result = client.login("user@example.com", "password")
        assert isinstance(result, TokenResponse)

    # After exiting context manager, client should be closed
    assert client._client.is_closed


def test_client_explicit_close(success_transport: httpx.MockTransport) -> None:
    """Test that client can be explicitly closed."""
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=success_transport)

    result = c.login("user@example.com", "password")
    assert isinstance(result, TokenResponse)
    assert not c._client.is_closed

    c.close()
    assert c._client.is_closed


def test_client_close_idempotent(success_transport: httpx.MockTransport) -> None:
    """Test that calling close() multiple times is safe."""
    c = Client(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.Client(transport=success_transport)

    c.close()
    c.close()  # Should not raise
    assert c._client.is_closed
