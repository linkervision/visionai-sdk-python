import pytest
import httpx
import jwt
from unittest.mock import patch

from visionai_sdk_python.async_client import AsyncClient
from visionai_sdk_python.exceptions import APIError, AuthenticationError, NetworkError, ServerError
from visionai_sdk_python.models import TokenResponse

AUTH_URL = "https://auth.example.com"
VLM_URL = "https://vlm.example.com"

TOKEN_PAYLOAD = {
    "access_token": "test-jwt-token",
    "expires_in": 3600,
    "token_type": "Bearer",
}


class AsyncMockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self._handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)  # noqa: RUF100


class AsyncRaisingTransport(httpx.AsyncBaseTransport):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def handle_async_request(self, _request: httpx.Request) -> httpx.Response:
        raise self._exc


@pytest.fixture
def success_transport() -> AsyncMockTransport:
    """Transport that returns successful token response."""
    return AsyncMockTransport(lambda r: httpx.Response(200, json=TOKEN_PAYLOAD))


@pytest.fixture
def unauthorized_transport() -> AsyncMockTransport:
    """Transport that returns 401 Unauthorized."""
    return AsyncMockTransport(
        lambda r: httpx.Response(401, json={"detail": "Invalid credentials"})
    )


@pytest.fixture
def server_error_transport() -> AsyncMockTransport:
    """Transport that returns 503 Service Unavailable."""
    return AsyncMockTransport(
        lambda r: httpx.Response(503, json={"detail": "Service Unavailable"})
    )


@pytest.fixture
def connect_error_transport() -> AsyncRaisingTransport:
    """Transport that raises ConnectError."""
    return AsyncRaisingTransport(httpx.ConnectError("Connection refused"))


@pytest.fixture
def api_error_transport() -> AsyncMockTransport:
    return AsyncMockTransport(
        lambda r: httpx.Response(310, json={"detail": "Too many redirect"})
    )


@pytest.fixture
def mock_client(success_transport: AsyncMockTransport) -> AsyncClient:
    """Create an AsyncClient with mocked successful transport."""
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=success_transport)
    return c


# --- login ---

@pytest.mark.asyncio
async def test_login_success(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides an AsyncClient backed by a 200 success transport

    # Act
    result = await mock_client.login("user@example.com", "correct-password")

    # Assert
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]
    assert result.expires_in == TOKEN_PAYLOAD["expires_in"]
    assert result.token_type == TOKEN_PAYLOAD["token_type"]


@pytest.mark.asyncio
async def test_login_wrong_credentials(unauthorized_transport: AsyncMockTransport) -> None:
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=unauthorized_transport)

    # Act & Assert
    with pytest.raises(AuthenticationError, match="Invalid credentials") as exc_info:
        await c.login("user@example.com", "wrong-password")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_login_server_error(server_error_transport: AsyncMockTransport) -> None:
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=server_error_transport)

    # Act & Assert
    with pytest.raises(ServerError, match="Service Unavailable") as exc_info:
        await c.login("user@example.com", "password")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_login_network_error(connect_error_transport: AsyncRaisingTransport) -> None:
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=connect_error_transport)

    # Act & Assert
    with pytest.raises(NetworkError, match="Network error"):
        await c.login("user@example.com", "password")


@pytest.mark.asyncio
async def test_login_api_error(api_error_transport):
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=api_error_transport)

    # Act & Assert
    with pytest.raises(APIError, match="Too many redirect") as exc_info:
        await c.login("user@example.com", "correct-password")
    assert exc_info.value.status_code == 310


@pytest.mark.asyncio
async def test_login_empty_email(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="email must not be empty"):
        await mock_client.login("", "password")


@pytest.mark.asyncio
async def test_login_empty_password(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="password must not be empty"):
        await mock_client.login("user@example.com", "")


@pytest.mark.asyncio
async def test_login_whitespace_email(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="email must not be empty"):
        await mock_client.login("   ", "password")


@pytest.mark.asyncio
async def test_login_whitespace_password(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="password must not be empty"):
        await mock_client.login("user@example.com", "   ")


# --- get_access_token ---

@pytest.mark.asyncio
async def test_get_access_token_success(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides an AsyncClient backed by a 200 success transport

    # Act
    result = await mock_client.get_access_token("admin", "secret")

    # Assert
    assert isinstance(result, TokenResponse)
    assert result.access_token == TOKEN_PAYLOAD["access_token"]


@pytest.mark.asyncio
async def test_get_access_token_wrong_credentials(unauthorized_transport: AsyncMockTransport) -> None:
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=unauthorized_transport)

    # Act & Assert
    with pytest.raises(AuthenticationError, match="Invalid credentials") as exc_info:
        await c.get_access_token("wrong-id", "wrong-secret")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_access_token_server_error(server_error_transport: AsyncMockTransport) -> None:
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=server_error_transport)

    # Act & Assert
    with pytest.raises(ServerError, match="Service Unavailable") as exc_info:
        await c.get_access_token("admin", "secret")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_get_access_token_network_error(connect_error_transport: AsyncRaisingTransport) -> None:
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=connect_error_transport)

    # Act & Assert
    with pytest.raises(NetworkError, match="Network error"):
        await c.get_access_token("admin", "secret")


@pytest.mark.asyncio
async def test_get_access_token_api_error(api_error_transport):
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=api_error_transport)

    # Act & Assert
    with pytest.raises(APIError, match="Too many redirect") as exc_info:
        await c.get_access_token("admin", "secret")
    assert exc_info.value.status_code == 310


@pytest.mark.asyncio
async def test_get_access_token_empty_client_id(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="client_id must not be empty"):
        await mock_client.get_access_token("", "secret")


@pytest.mark.asyncio
async def test_get_access_token_empty_client_secret(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        await mock_client.get_access_token("client-id", "")


@pytest.mark.asyncio
async def test_get_access_token_whitespace_client_id(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="client_id must not be empty"):
        await mock_client.get_access_token("   ", "secret")


@pytest.mark.asyncio
async def test_get_access_token_whitespace_client_secret(mock_client: AsyncClient) -> None:
    # Arrange: mock_client fixture provides a ready-to-use AsyncClient

    # Act & Assert
    with pytest.raises(ValueError, match="client_secret must not be empty"):
        await mock_client.get_access_token("client-id", "   ")


# --- is_token_valid ---
@pytest.mark.asyncio
async def test_is_token_valid_returns_true_on_valid_token(mock_client: AsyncClient) -> None:
    # Arrange
    claims = {"sub": "user-123", "iss": "https://auth.example.com"}
    with patch.object(mock_client._jwt_verifier, "verify_async", return_value=claims):
        # Act
        result = await mock_client.is_token_valid("any.jwt.token")

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
async def test_is_token_valid_logs_warning_on_invalid_token(
    mock_client: AsyncClient, exc: jwt.InvalidTokenError, exc_name: str, caplog
) -> None:
    # Arrange
    from visionai_sdk_python import async_client
    with patch.object(mock_client._jwt_verifier, "verify_async", side_effect=exc):
        # Act
        with caplog.at_level("WARNING", logger=async_client.__name__):
            result = await mock_client.is_token_valid("any.jwt.token")

    # Assert
    assert result is False
    assert len(caplog.records) == 1

    log_record = caplog.records[0]
    assert log_record.levelname == "WARNING"
    assert "Token validation failed" in log_record.message
    assert exc_name in log_record.message
    assert log_record.jwt_error_type == exc_name
    assert log_record.jwt_error_message == str(exc)


@pytest.mark.asyncio
async def test_is_token_valid_logs_error_on_jwks_failure(
    mock_client: AsyncClient, caplog
) -> None:
    # Arrange
    from visionai_sdk_python import async_client
    exc = jwt.PyJWKClientError("Failed to fetch JWKS from endpoint")
    with patch.object(mock_client._jwt_verifier, "verify_async", side_effect=exc):
        # Act
        with caplog.at_level("ERROR", logger=async_client.__name__):
            result = await mock_client.is_token_valid("any.jwt.token")

    # Assert
    assert result is False
    assert len(caplog.records) == 1

    log_record = caplog.records[0]
    assert log_record.levelname == "ERROR"
    assert "JWKS client error during token validation" in log_record.message
    assert log_record.jwt_error_type == "PyJWKClientError"
    assert log_record.jwt_error_message == str(exc)


@pytest.mark.asyncio
async def test_is_token_valid_propagates_unexpected_errors(
    mock_client: AsyncClient
) -> None:
    # Arrange - Simulate a programming error (e.g., AttributeError)
    exc = AttributeError("'NoneType' object has no attribute 'verify'")
    with patch.object(mock_client._jwt_verifier, "verify_async", side_effect=exc):
        # Act & Assert - Unexpected exceptions should propagate
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'verify'"):
            await mock_client.is_token_valid("any.jwt.token")


# --- Resource cleanup ---

@pytest.mark.asyncio
async def test_async_client_context_manager(success_transport: AsyncMockTransport) -> None:
    """Test that async client properly closes resources when used as context manager."""
    # Arrange
    async with AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL) as client:
        client._client = httpx.AsyncClient(transport=success_transport)

        # Act
        result = await client.login("user@example.com", "password")

        # Assert: login returned a valid token while the client is open
        assert isinstance(result, TokenResponse)

    # Assert: underlying HTTP client is closed after exiting the context manager
    assert client._client.is_closed


@pytest.mark.asyncio
async def test_async_client_explicit_close(success_transport: AsyncMockTransport) -> None:
    """Test that async client can be explicitly closed."""
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=success_transport)

    # Act
    result = await c.login("user@example.com", "password")

    # Assert: client is still open after a successful call
    assert isinstance(result, TokenResponse)
    assert not c._client.is_closed

    # Act: explicitly close the client
    await c.close()

    # Assert: underlying HTTP client is now closed
    assert c._client.is_closed


@pytest.mark.asyncio
async def test_async_client_close_idempotent(success_transport: AsyncMockTransport) -> None:
    """Test that calling close() multiple times is safe."""
    # Arrange
    c = AsyncClient(auth_url=AUTH_URL, vlm_url=VLM_URL)
    c._client = httpx.AsyncClient(transport=success_transport)

    # Act: close twice to verify idempotency
    await c.close()
    await c.close()  # Should not raise

    # Assert
    assert c._client.is_closed
