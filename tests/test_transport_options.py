"""An injected http client brings its own transport settings; the SDK refuses to take more."""

import httpx
import pytest

from visionai_sdk_python import (
    AsyncClient,
    AsyncClientCredentials,
    Client,
    ClientCredentials,
)


def _sync_http() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))


def _async_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: Client("http://a", http_client=_sync_http(), timeout=1.0),
        lambda: Client("http://a", http_client=_sync_http(), verify_ssl=False),
        lambda: Client("http://a", http_client=_sync_http(), max_connections=5),
        lambda: AsyncClient(
            "http://a", http_client=_async_http(), max_keepalive_connections=5
        ),
        lambda: ClientCredentials(
            "http://a", "id", "s", http=_sync_http(), timeout=1.0
        ),
        lambda: AsyncClientCredentials(
            "http://a", "id", "s", http=_async_http(), verify_ssl=False
        ),
    ],
)
def test_transport_options_are_refused_with_an_injected_client(build):
    with pytest.raises(ValueError, match="has its own verify, timeout and limits"):
        build()


def test_defaults_apply_when_the_sdk_builds_the_client():
    client = Client("http://a")

    assert (client.verify_ssl, client.timeout) == (True, 10.0)
    assert (client.max_connections, client.max_keepalive_connections) == (100, 20)
    assert Client("http://a", timeout=3.0, verify_ssl=False).timeout == 3.0


def test_an_injected_client_is_used_as_is():
    http = _sync_http()

    assert Client("http://a", http_client=http)._client is http
