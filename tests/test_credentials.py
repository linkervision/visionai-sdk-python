import asyncio

import httpx
import pytest

from tests.constants import (
    AUTH_URL,
    NORMAL_COMPLETED_RESPONSE,
    TOKEN_PAYLOAD,
    VALID_NIM_PAYLOAD,
    VLM_URL,
)
from visionai_sdk_python import (
    AccessToken,
    AsyncClient,
    AsyncClientCredentials,
    Client,
    ClientCredentials,
    RateLimitedError,
)
from visionai_sdk_python.auth import credentials as credentials_module
from visionai_sdk_python.auth.credentials import (
    QUOTA_COOLDOWN_S,
    REFRESH_BUFFER_S,
)
from visionai_sdk_python.exceptions import (
    AuthenticationError,
    NetworkError,
    VisionaiSDKError,
)

EXPIRES_IN = 259200  # Platform issues 3-day tokens


class FakePlatform:
    """The client-token endpoint: numbered tokens, or a forced status / body."""

    def __init__(self) -> None:
        self.minted = 0
        self.requests: list[httpx.Request] = []
        self.status: int | None = None
        self.body: object = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status is not None:
            return httpx.Response(self.status, json={"detail": "forced"})
        if self.body is not None:
            return httpx.Response(200, json=self.body)
        self.minted += 1
        return httpx.Response(
            200,
            json={
                **TOKEN_PAYLOAD,
                "access_token": f"tok-{self.minted}",
                "expires_in": EXPIRES_IN,
            },
        )


@pytest.fixture
def platform() -> FakePlatform:
    return FakePlatform()


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    """The credential's clock; tests move it by editing clock[0]."""
    now = [1_000_000.0]
    monkeypatch.setattr(credentials_module, "_now", lambda: now[0])
    return now


@pytest.fixture
def cred(platform, clock) -> AsyncClientCredentials:
    return AsyncClientCredentials(
        AUTH_URL,
        "cid",
        "secret",
        http=httpx.AsyncClient(transport=httpx.MockTransport(platform.handler)),
    )


async def test_get_token_posts_the_client_credentials_once_and_caches(platform, cred):
    first = await cred.get_token()
    second = await cred.get_token()

    assert first == second == AccessToken("tok-1", 1_000_000.0 + EXPIRES_IN)
    assert platform.minted == 1
    (request,) = platform.requests
    assert str(request.url) == f"{AUTH_URL}/api/users/client-token"
    assert (
        b'"client_id":"cid"' in request.content
        or b'"client_id": "cid"' in request.content
    )
    assert cred.last_response is not None
    assert cred.last_response.token_type == "Bearer"


async def test_refresh_starts_five_minutes_before_expiry(platform, cred, clock):
    await cred.get_token()

    clock[0] += EXPIRES_IN - REFRESH_BUFFER_S - 1
    assert (await cred.get_token()).token == "tok-1"
    clock[0] += 2
    assert (await cred.get_token()).token == "tok-2"
    assert platform.minted == 2


async def test_concurrent_callers_share_one_mint(platform, cred):
    tokens = await asyncio.gather(*(cred.get_token() for _ in range(10)))

    assert {t.token for t in tokens} == {"tok-1"}
    assert platform.minted == 1


async def test_a_401_buys_one_replacement_and_a_rejected_replacement_is_kept(
    platform, cred, clock, caplog
):
    await cred.get_token()
    assert cred.invalidate("tok-1") is True
    assert (await cred.get_token()).token == "tok-2"

    # A late 401 for the old token changes nothing.
    assert cred.invalidate("tok-1") is False
    assert (await cred.get_token()).token == "tok-2"
    # The replacement rejected too: a client problem, not expiry. Keep it.
    assert cred.invalidate("tok-2") is False
    assert (await cred.get_token()).token == "tok-2"
    assert platform.minted == 2
    assert "replacement token rejected too" in caplog.text

    # Its natural refresh resets the rule.
    clock[0] += EXPIRES_IN
    assert (await cred.get_token()).token == "tok-3"
    assert cred.invalidate("tok-3") is True


async def test_a_429_stops_minting_for_an_hour(platform, cred, clock):
    platform.status = 429
    with pytest.raises(RateLimitedError):
        await cred.get_token()

    platform.status = None
    with pytest.raises(RateLimitedError, match="cooldown"):
        await cred.get_token()
    assert len(platform.requests) == 1

    clock[0] += QUOTA_COOLDOWN_S + 1
    assert (await cred.get_token()).token == "tok-1"


async def test_platform_refusing_the_client_is_an_authentication_error(platform, cred):
    platform.status = 401

    with pytest.raises(AuthenticationError):
        await cred.get_token()


@pytest.mark.parametrize(
    "body",
    [
        {**TOKEN_PAYLOAD, "access_token": None},
        {**TOKEN_PAYLOAD, "access_token": ""},
        {**TOKEN_PAYLOAD, "expires_in": None},
        {"detail": "no token here"},
        ["not", "an", "object"],
    ],
)
async def test_a_body_without_a_usable_token_is_an_sdk_error(platform, cred, body):
    platform.body = body

    with pytest.raises(VisionaiSDKError, match="not a token"):
        await cred.get_token()


async def test_an_unreachable_platform_is_a_network_error(clock):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    cred = AsyncClientCredentials(
        AUTH_URL,
        "cid",
        "secret",
        http=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )

    with pytest.raises(NetworkError):
        await cred.get_token()


@pytest.mark.parametrize(
    ("auth_url", "client_id", "client_secret"),
    [("", "cid", "secret"), (AUTH_URL, " ", "secret"), (AUTH_URL, "cid", "")],
)
def test_empty_arguments_are_rejected(auth_url, client_id, client_secret):
    with pytest.raises(ValueError, match="must not be empty"):
        ClientCredentials(auth_url, client_id, client_secret)


def test_sync_credentials_cache_and_replace_once(platform, clock):
    with ClientCredentials(
        AUTH_URL,
        "cid",
        "secret",
        http=httpx.Client(transport=httpx.MockTransport(platform.handler)),
    ) as cred:
        assert cred.get_token().token == "tok-1"
        assert cred.get_token().token == "tok-1"
        assert cred.invalidate("tok-1") is True
        assert cred.get_token().token == "tok-2"
        assert cred.invalidate("tok-2") is False
        assert cred.get_token().token == "tok-2"

    assert platform.minted == 2


class RecordingTransport(httpx.AsyncBaseTransport):
    """A VLM that records the Authorization header and answers a completed chat."""

    def __init__(self) -> None:
        self.authorizations: list[str | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.authorizations.append(request.headers.get("authorization"))
        return httpx.Response(200, json=NORMAL_COMPLETED_RESPONSE)


async def test_async_client_calls_vlm_with_the_injected_credential(cred, clock):
    vlm = RecordingTransport()
    client = AsyncClient(AUTH_URL, VLM_URL, credential=cred)
    client._client = httpx.AsyncClient(transport=vlm)

    await client.vlm.chat(VALID_NIM_PAYLOAD)
    clock[0] += EXPIRES_IN
    await client.vlm.chat(VALID_NIM_PAYLOAD)

    assert vlm.authorizations == ["Bearer tok-1", "Bearer tok-2"]


async def test_a_client_without_vlm_url_can_authenticate_but_not_chat(cred):
    client = AsyncClient(AUTH_URL, credential=cred)

    await client._ensure_token()
    assert client._access_token == "tok-1"
    with pytest.raises(ValueError, match="vlm_url is required"):
        await client.vlm.chat(VALID_NIM_PAYLOAD)


async def test_get_access_token_keeps_its_result_and_refreshes_through_a_credential(
    platform, clock
):
    client = AsyncClient(AUTH_URL, VLM_URL)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(platform.handler))

    response = await client.auth.get_access_token("cid", "secret")

    assert (response.access_token, response.expires_in) == ("tok-1", EXPIRES_IN)
    assert client._access_token == "tok-1"
    assert client._credentials == {"client_id": "cid", "client_secret": "secret"}
    await client._ensure_token()
    assert platform.minted == 1
    clock[0] += EXPIRES_IN
    await client._ensure_token()
    assert client._access_token == "tok-2"


def test_sync_get_access_token_keeps_its_result(platform, clock):
    client = Client(AUTH_URL, VLM_URL)
    client._client = httpx.Client(transport=httpx.MockTransport(platform.handler))

    response = client.auth.get_access_token("cid", "secret")

    assert response.access_token == "tok-1"
    client._ensure_token()
    assert client._access_token == "tok-1"
    assert platform.minted == 1
