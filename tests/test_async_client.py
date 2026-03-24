import pytest
import httpx

from visionai_sdk_python.async_client import AsyncClient
from visionai_sdk_python.exceptions import AuthenticationError, NetworkError, ServerError
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
