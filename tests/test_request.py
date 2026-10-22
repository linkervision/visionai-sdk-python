"""``client.request()``: any URL with the client's token, one retry on 401."""

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
    AsyncClient,
    AsyncClientCredentials,
    Client,
    ClientCredentials,
)
from visionai_sdk_python.auth import credentials as credentials_module
from visionai_sdk_python.auth.credentials import REPLACE_COOLDOWN_S
from visionai_sdk_python.exceptions import AuthenticationError
from visionai_sdk_python.vlm.models import ResponseNormalModel

NODES = "https://vs.example.com/mgmt/nodes"


class FakeWorld:
    """Platform (mint, login) and token-checking services behind one handler."""

    def __init__(self) -> None:
        self.minted = 0
        self.logins = 0
        self.rejected: set[str] = set()
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/users/client-token":
            self.minted += 1
            token = f"tok-{self.minted}"
            return httpx.Response(200, json={**TOKEN_PAYLOAD, "access_token": token})
        if path == "/api/users/jwt":
            self.logins += 1
            token = f"login-{self.logins}"
            return httpx.Response(200, json={**TOKEN_PAYLOAD, "access_token": token})
        self.calls.append(request)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not token or token in self.rejected:
            return httpx.Response(401, json={"detail": "invalid token"})
        if path == "/api/chat":
            return httpx.Response(200, json=NORMAL_COMPLETED_RESPONSE)
        return httpx.Response(200, json={"path": path})

    def tokens_sent(self) -> list[str]:
        return [
            r.headers.get("authorization", "").removeprefix("Bearer ")
            for r in self.calls
        ]


@pytest.fixture
def world() -> FakeWorld:
    return FakeWorld()


@pytest.fixture
def transport(world) -> httpx.MockTransport:
    return httpx.MockTransport(world.handler)


@pytest.fixture
def client(transport) -> AsyncClient:
    cred = AsyncClientCredentials(
        AUTH_URL, "cid", "secret", http=httpx.AsyncClient(transport=transport)
    )
    return AsyncClient(
        AUTH_URL,
        VLM_URL,
        credential=cred,
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    """The credential's clock; tests move it by editing clock[0]."""
    now = [1_000_000.0]
    monkeypatch.setattr(credentials_module, "_now", lambda: now[0])
    return now


@pytest.fixture
async def aged_client(client, clock) -> AsyncClient:
    """A client whose token is old enough for a 401 to replace it."""
    await client.credential.get_token()
    clock[0] += REPLACE_COOLDOWN_S + 1
    return client


@pytest.fixture
def sync_client(transport) -> Client:
    cred = ClientCredentials(
        AUTH_URL, "cid", "secret", http=httpx.Client(transport=transport)
    )
    return Client(
        AUTH_URL,
        VLM_URL,
        credential=cred,
        http_client=httpx.Client(transport=transport),
    )


@pytest.fixture
def aged_sync_client(sync_client, clock) -> Client:
    sync_client.credential.get_token()
    clock[0] += REPLACE_COOLDOWN_S + 1
    return sync_client


async def test_an_injected_http_client_is_used_and_not_closed(transport):
    http = httpx.AsyncClient(transport=transport)
    async with AsyncClient(AUTH_URL, http_client=http) as client:
        assert client._client is http
    assert not http.is_closed

    owned = AsyncClient(AUTH_URL)
    await owned.close()
    assert owned._client.is_closed


def test_the_credential_is_reachable_for_callers_that_need_the_raw_token(client):
    assert isinstance(client.credential, AsyncClientCredentials)
    assert AsyncClient(AUTH_URL).credential is None


async def test_request_sends_the_token_to_any_url(world, client):
    response = await client.request("GET", NODES)

    assert response.status_code == 200
    assert world.tokens_sent() == ["tok-1"]
    assert str(world.calls[0].url) == NODES


async def test_request_refuses_a_bare_path(world, client):
    with pytest.raises(ValueError, match="full URL"):
        await client.request("GET", "/api/anything")

    assert world.calls == []


async def test_request_keeps_caller_headers_but_uses_the_clients_token(world, client):
    await client.request(
        "GET",
        NODES,
        headers={"X-Request-Source": "agent", "Authorization": "Bearer nope"},
    )

    (call,) = world.calls
    assert call.headers["x-request-source"] == "agent"
    assert call.headers["authorization"] == "Bearer tok-1"


async def test_request_retries_once_with_a_fresh_token_after_a_401(world, aged_client):
    world.rejected = {"tok-1"}

    response = await aged_client.request("GET", NODES)

    assert response.status_code == 200
    assert world.tokens_sent() == ["tok-1", "tok-2"]
    assert world.minted == 2


async def test_request_does_not_mint_a_third_token_when_the_replacement_is_rejected(
    world, aged_client
):
    world.rejected = {"tok-1", "tok-2"}

    for _ in range(3):
        with pytest.raises(AuthenticationError):
            await aged_client.request("GET", NODES)

    assert world.minted == 2
    assert world.tokens_sent() == ["tok-1", "tok-2", "tok-2", "tok-2"]


async def test_request_after_login_raises_on_401_without_a_retry(world, transport):
    # The 401 rule lives in the credential; a login() token has none.
    client = AsyncClient(
        AUTH_URL, VLM_URL, http_client=httpx.AsyncClient(transport=transport)
    )
    await client.auth.login("user@example.com", "password")
    world.rejected = {"login-1"}

    with pytest.raises(AuthenticationError):
        await client.request("GET", NODES)

    assert world.logins == 1
    assert world.tokens_sent() == ["login-1"]


async def test_request_with_an_injected_token_and_no_credentials_raises_on_401(
    world, transport
):
    # Dataverse's path: a token it minted itself, nothing to refresh with.
    client = AsyncClient(
        AUTH_URL, VLM_URL, http_client=httpx.AsyncClient(transport=transport)
    )
    client._store_token("fixed", expires_in=3600)
    world.rejected = {"fixed"}

    with pytest.raises(AuthenticationError):
        await client.request("GET", NODES)

    assert world.tokens_sent() == ["fixed"]
    assert world.minted == 0


async def test_chat_gets_one_fresh_token_on_401(world, aged_client):
    world.rejected = {"tok-1"}

    result = await aged_client.vlm.chat(VALID_NIM_PAYLOAD)

    assert isinstance(result, ResponseNormalModel)
    assert result.status == "completed"
    assert world.tokens_sent() == ["tok-1", "tok-2"]


def test_sync_request_sends_the_token_and_retries_once(world, aged_sync_client):
    world.rejected = {"tok-1"}

    response = aged_sync_client.request("GET", NODES)

    assert response.status_code == 200
    assert world.tokens_sent() == ["tok-1", "tok-2"]


def test_sync_request_keeps_a_rejected_replacement(world, aged_sync_client):
    world.rejected = {"tok-1", "tok-2"}

    with pytest.raises(AuthenticationError):
        aged_sync_client.request("GET", NODES)
    with pytest.raises(AuthenticationError):
        aged_sync_client.vlm.chat(VALID_NIM_PAYLOAD)

    assert world.minted == 2


async def test_a_401_on_a_token_minted_moments_ago_is_not_retried(world, client, clock):
    world.rejected = {"tok-1"}

    with pytest.raises(AuthenticationError):
        await client.request("GET", NODES)
    assert world.minted == 1
    assert world.tokens_sent() == ["tok-1"]

    clock[0] += REPLACE_COOLDOWN_S + 1
    assert (await client.request("GET", NODES)).status_code == 200
    assert world.minted == 2


async def test_concurrent_401s_share_one_replacement_and_all_retry(world, clock):
    class Yielding(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0)  # every caller sends before any 401 is handled
            return world.handler(request)

    cred = AsyncClientCredentials(
        AUTH_URL, "cid", "secret", http=httpx.AsyncClient(transport=Yielding())
    )
    client = AsyncClient(
        AUTH_URL,
        VLM_URL,
        credential=cred,
        http_client=httpx.AsyncClient(transport=Yielding()),
    )
    await cred.get_token()
    clock[0] += REPLACE_COOLDOWN_S + 1
    world.rejected = {"tok-1"}

    responses = await asyncio.gather(*(client.request("GET", NODES) for _ in range(5)))

    assert [r.status_code for r in responses] == [200] * 5
    assert world.minted == 2
    assert sorted(world.tokens_sent()) == ["tok-1"] * 5 + ["tok-2"] * 5


async def test_a_lowercase_authorization_header_is_replaced_not_duplicated(
    world, client
):
    await client.request("GET", NODES, headers={"authorization": "Bearer caller"})

    assert world.calls[-1].headers.get_list("authorization") == ["Bearer tok-1"]


async def test_a_streamed_body_is_not_retried_after_a_401(world, aged_client):
    world.rejected = {"tok-1"}

    async def body():
        yield b"part"

    with pytest.raises(AuthenticationError):
        await aged_client.request("POST", NODES, content=body())

    assert world.minted == 1
    assert world.tokens_sent() == ["tok-1"]
