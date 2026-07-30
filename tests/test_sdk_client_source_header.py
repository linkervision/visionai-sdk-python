"""The SDK's own clients must attribute themselves, not just the shims.

Client/AsyncClient previously built a bare httpx client with no headers, so a
service that migrated onto the SDK produced no attribution at all — the
opposite of what AB#42883 is for.
"""

import http.server
import socketserver
import threading

import pytest

from visionai_sdk_python import AsyncClient, Client
from visionai_sdk_python._source_header import SOURCE_ENV_VAR, SOURCE_HEADER

MISSING = "MISSING"


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = (self.headers.get(SOURCE_HEADER) or MISSING).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def url():
    server = socketserver.TCPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


class TestDefaultHeaderOnSdkClients:
    def test_sync_client_sets_default_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        client = Client(auth_url="http://a.test", vlm_url="http://v.test")

        assert client._client.headers[SOURCE_HEADER] == "stream-agent"

    def test_async_client_sets_default_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        client = AsyncClient(auth_url="http://a.test", vlm_url="http://v.test")

        assert client._client.headers[SOURCE_HEADER] == "stream-agent"

    def test_no_header_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)

        client = Client(auth_url="http://a.test", vlm_url="http://v.test")

        assert SOURCE_HEADER not in client._client.headers

    def test_malformed_value_is_dropped_not_fatal(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream\nagent")

        with pytest.warns(RuntimeWarning):
            client = Client(auth_url="http://a.test", vlm_url="http://v.test")

        assert SOURCE_HEADER not in client._client.headers

    def test_header_reaches_the_wire(self, url, monkeypatch):
        """Asserted end-to-end, not just on the client object."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        client = Client(auth_url=url, vlm_url=url)

        assert client._request("GET", url).text == "stream-agent"

    def test_trailing_newline_is_stripped(self, url, monkeypatch):
        """Helm block scalars produce a trailing newline."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent\n")

        client = Client(auth_url=url, vlm_url=url)

        assert client._request("GET", url).text == "stream-agent"
