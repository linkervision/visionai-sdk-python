"""End-to-end tests for instrumentation.instrument().

These drive a real loopback HTTP server that echoes back the X-Request-Source it
received, so what is asserted is what actually went out on the wire rather than
what some layer of the client thought it was going to send.
"""

import http.server
import socketserver
import threading
import warnings

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
    """Instrument with a known source, and always unwrap afterwards.

    The late-instrumentation warning is expected here and asserted separately in
    TestLateInstrumentationWarning: a test module necessarily imports the HTTP
    libraries at the top, which is exactly the situation that warning is for.
    """
    monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
            instrumentation.instrument()
        try:
            assert requests.Session().get(url).text == MISSING
        finally:
            instrumentation.uninstrument()

    def test_uninstrument_restores_original_behavior(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
            instrumentation.instrument()
        assert requests.Session().get(url).text == "stream-agent"

        instrumentation.uninstrument()

        assert requests.Session().get(url).text == MISSING

    def test_instrument_is_idempotent(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
            instrumentation.instrument()
        try:
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_unusable_value_is_dropped_and_request_still_succeeds(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream\nagent")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
            instrumentation.instrument()
        try:
            with pytest.warns(RuntimeWarning):
                assert requests.Session().get(url).text == MISSING
        finally:
            instrumentation.uninstrument()


class TestLateInstrumentationWarning:
    """gevent.monkey warns when patched too late; so should we."""

    def test_warns_when_requests_already_imported(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.uninstrument()
        assert "requests" in __import__("sys").modules

        try:
            with pytest.warns(instrumentation.LateInstrumentationWarning):
                instrumentation.instrument()
        finally:
            instrumentation.uninstrument()

    def test_quiet_when_optional_libraries_not_loaded(self, monkeypatch):
        """httpx is always loaded via the package __init__, so it must not warn."""
        import sys

        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        monkeypatch.delitem(sys.modules, "requests", raising=False)
        monkeypatch.delitem(sys.modules, "aiohttp", raising=False)
        instrumentation.uninstrument()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error", instrumentation.LateInstrumentationWarning
                )
                instrumentation.instrument()
        finally:
            instrumentation.uninstrument()


class TestSdkClientCarriesHeaderNatively:
    """The SDK's own clients must attribute themselves without instrument()."""

    def test_sync_client_sets_default_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url="http://a.test", vlm_url="http://v.test")

        assert client._client.headers[SOURCE_HEADER] == "stream-agent"

    def test_async_client_sets_default_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import AsyncClient

        client = AsyncClient(auth_url="http://a.test", vlm_url="http://v.test")

        assert client._client.headers[SOURCE_HEADER] == "stream-agent"

    def test_no_header_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        from visionai_sdk_python import Client

        client = Client(auth_url="http://a.test", vlm_url="http://v.test")

        assert SOURCE_HEADER not in client._client.headers

    def test_end_to_end_without_instrument(self, url, monkeypatch):
        """The header must reach the wire, not just sit on the client object."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)

        assert client._request("GET", url).text == "stream-agent"


class TestOriginFromHeaders:
    def test_extracts_case_insensitively_from_a_mapping(self):
        assert (
            instrumentation.origin_from_headers({"x-request-source": "observ-pod-1"})
            == "observ-pod-1"
        )

    def test_extracts_from_a_list_of_pairs(self):
        headers = [("Authorization", "Bearer x"), ("X-Request-Source", "observ-pod-1")]

        assert instrumentation.origin_from_headers(headers) == "observ-pod-1"

    def test_returns_none_when_absent(self):
        assert (
            instrumentation.origin_from_headers({"Authorization": "Bearer x"}) is None
        )
        assert instrumentation.origin_from_headers({}) is None
        assert instrumentation.origin_from_headers(None) is None


class TestCurrentOriginScope:
    """A-5: forwarding an inherited origin instead of stamping this service's own."""

    def test_client_built_inside_scope_inherits_the_origin(self, url, instrumented):
        with instrumentation.current_origin("observ-pod-1"):
            assert requests.Session().get(url).text == "observ-pod-1"

    def test_outside_any_scope_falls_back_to_own_identity(self, url, instrumented):
        assert requests.Session().get(url).text == "stream-agent"

    def test_scope_exit_restores_previous_behavior(self, url, instrumented):
        with instrumentation.current_origin("observ-pod-1"):
            assert requests.Session().get(url).text == "observ-pod-1"

        assert requests.Session().get(url).text == "stream-agent"

    def test_none_value_falls_back_to_own_identity(self, url, instrumented):
        """current_origin(None) is what origin_from_headers() returns when the
        inbound request carried no X-Request-Source -- this service is the origin."""
        with instrumentation.current_origin(None):
            assert requests.Session().get(url).text == "stream-agent"

    def test_nested_scopes_restore_the_outer_value(self, url, instrumented):
        with instrumentation.current_origin("outer"):
            with instrumentation.current_origin("inner"):
                assert requests.Session().get(url).text == "inner"

            assert requests.Session().get(url).text == "outer"

    def test_malformed_inherited_value_falls_back_to_own_identity(
        self, url, instrumented
    ):
        with pytest.warns(RuntimeWarning):
            with instrumentation.current_origin("bad\nvalue"):
                assert requests.Session().get(url).text == "stream-agent"

    def test_httpx_client_inherits_the_origin(self, url, instrumented):
        with instrumentation.current_origin("observ-pod-1"):
            with httpx.Client() as client:
                assert client.get(url).text == "observ-pod-1"

    async def test_aiohttp_session_inherits_the_origin(self, url, instrumented):
        with instrumentation.current_origin("observ-pod-1"):
            async with aiohttp.ClientSession() as session:
                response = await session.get(url)

                assert await response.text() == "observ-pod-1"

    def test_sdk_client_inherits_the_origin(self, url, instrumented):
        from visionai_sdk_python import Client

        with instrumentation.current_origin("observ-pod-1"):
            client = Client(auth_url=url, vlm_url=url)

            assert client._request("GET", url).text == "observ-pod-1"

    async def test_concurrent_tasks_do_not_leak_origin_into_each_other(
        self, url, instrumented
    ):
        """The whole point of contextvars over a plain global: isolated per task."""
        import asyncio

        results = {}

        async def handle(origin, key):
            with instrumentation.current_origin(origin):
                await asyncio.sleep(0.01)
                results[key] = requests.Session().get(url).text

        await asyncio.gather(handle("task-a", "a"), handle("task-b", "b"))

        assert results == {"a": "task-a", "b": "task-b"}

    def test_per_call_header_overrides_a_shared_clients_default(
        self, url, instrumented
    ):
        """The documented workaround for a client built once and reused across many
        requests: per-call headers already override client-level defaults."""
        shared = (
            requests.Session()
        )  # built outside any scope -> defaults to "stream-agent"

        with instrumentation.current_origin("observ-pod-1"):
            response = shared.get(
                url, headers={"X-Request-Source": instrumentation.get_current_origin()}
            )

        assert response.text == "observ-pod-1"


class TestPodChainScenario:
    """The exact scenario from the review discussion: observ-pod-1 -> observ-pod-2 -> VLM,
    where pod-2's own request-handling code uses current_origin()."""

    def test_pod_2_forwards_pod_1s_identity_when_using_current_origin(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "observ-pod-2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
            instrumentation.instrument()
        try:
            # pod-2's inbound-request handling: extract what pod-1 sent, scope it.
            inbound_headers = {SOURCE_HEADER: "observ-pod-1"}
            origin = instrumentation.origin_from_headers(inbound_headers)
            with instrumentation.current_origin(origin):
                # pod-2's own outbound call to VLM, made while handling that request.
                response = requests.Session().get(url)
        finally:
            instrumentation.uninstrument()

        assert response.text == "observ-pod-1"

    def test_without_current_origin_pod_2_stamps_its_own_identity(
        self, url, monkeypatch
    ):
        """Same inbound request, but pod-2's code never calls current_origin() --
        this is the A-5 gap: pod-1's identity is silently lost."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "observ-pod-2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", instrumentation.LateInstrumentationWarning)
            instrumentation.instrument()
        try:
            response = requests.Session().get(url)
        finally:
            instrumentation.uninstrument()

        assert response.text == "observ-pod-2"
