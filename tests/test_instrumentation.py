"""End-to-end tests for instrumentation.instrument().

These drive a real loopback HTTP server that echoes back the X-Request-Source it
received, so what is asserted is what actually went out on the wire rather than
what some layer of the client thought it was going to send.
"""

import http.server
import socketserver
import threading

import aiohttp
import httpx
import pytest
import requests

from visionai_sdk_python import instrumentation
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


@pytest.fixture
def instrumented(monkeypatch):
    """Instrument with a known source, and always unwrap afterwards."""
    monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
    instrumentation.instrument()
    yield
    instrumentation.uninstrument()


class TestRequests:
    def test_session_carries_header(self, url, instrumented):
        assert requests.Session().get(url).text == "stream-agent"

    def test_module_level_one_shot_carries_header(self, url, instrumented):
        """requests.get() builds a Session internally, so it is covered too."""
        assert requests.get(url).text == "stream-agent"

    def test_send_prepared_request_carries_header(self, url, instrumented):
        session = requests.Session()
        prepared = session.prepare_request(requests.Request("GET", url))

        assert session.send(prepared).text == "stream-agent"

    def test_caller_supplied_value_wins(self, url, instrumented):
        session = requests.Session()

        assert session.get(url, headers={SOURCE_HEADER: "caller"}).text == "caller"

    def test_real_class_is_not_replaced(self, instrumented):
        assert requests.Session is requests.sessions.Session
        assert requests.exceptions.ConnectionError is not None


class TestHttpx:
    def test_client_carries_header(self, url, instrumented):
        with httpx.Client() as client:
            assert client.get(url).text == "stream-agent"

    def test_module_level_one_shot_carries_header(self, url, instrumented):
        assert httpx.get(url).text == "stream-agent"

    def test_send_prebuilt_request_carries_header(self, url, instrumented):
        with httpx.Client() as client:
            response = client.send(client.build_request("GET", url))

        assert response.text == "stream-agent"

    async def test_async_client_carries_header(self, url, instrumented):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        assert response.text == "stream-agent"

    def test_real_class_is_not_replaced(self, instrumented):
        assert httpx.Client.__mro__[0] is httpx.Client


class TestAiohttp:
    async def test_session_carries_header(self, url, instrumented):
        async with aiohttp.ClientSession() as session:
            response = await session.get(url)

            assert await response.text() == "stream-agent"

    async def test_caller_supplied_value_wins(self, url, instrumented):
        async with aiohttp.ClientSession() as session:
            response = await session.get(url, headers={SOURCE_HEADER: "caller"})

            assert await response.text() == "caller"

    async def test_isinstance_still_holds_and_no_deprecation_warning(
        self, url, instrumented
    ):
        """The shim's ClientSession subclass triggered aiohttp's subclassing warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            async with aiohttp.ClientSession() as session:
                assert isinstance(session, aiohttp.ClientSession)
                assert type(session) is aiohttp.ClientSession

    async def test_preserves_caller_duplicate_headers(self, url, instrumented):
        """Injecting as a pair list must not collapse a caller's duplicate headers."""
        from multidict import CIMultiDict

        headers = CIMultiDict([("X-Foo", "1"), ("X-Foo", "2")])
        async with aiohttp.ClientSession(headers=headers) as session:
            assert len(session.headers.getall("X-Foo")) == 2
            assert session.headers[SOURCE_HEADER] == "stream-agent"


class TestNoOpByDefault:
    def test_no_env_var_means_no_header(self, url, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        instrumentation.instrument()
        try:
            assert requests.Session().get(url).text == MISSING
        finally:
            instrumentation.uninstrument()

    def test_uninstrument_restores_original_behavior(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument()
        assert requests.Session().get(url).text == "stream-agent"

        instrumentation.uninstrument()

        assert requests.Session().get(url).text == MISSING

    def test_instrument_is_idempotent(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument()
        instrumentation.instrument()
        instrumentation.instrument()
        try:
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

        assert requests.Session().get(url).text == MISSING


class TestMalformedEnvVar:
    """A bad env var must not be able to break the caller's traffic."""

    def test_trailing_newline_is_stripped(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent\n")
        instrumentation.instrument()
        try:
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_unusable_value_is_dropped_and_request_still_succeeds(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream\nagent")
        instrumentation.instrument()
        try:
            with pytest.warns(RuntimeWarning):
                assert requests.Session().get(url).text == MISSING
        finally:
            instrumentation.uninstrument()
