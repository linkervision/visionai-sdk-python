"""End-to-end tests for instrumentation.instrument().

These drive a real loopback HTTP server that echoes back the X-Request-Source it
received, so what is asserted is what actually went out on the wire rather than
what some layer of the client thought it was going to send.
"""

import builtins
import http.server
import socketserver
import sys
import threading
import time
import urllib.parse
import warnings

import aiohttp
import httpx
import pytest
import requests

from visionai_sdk_python import instrumentation
from visionai_sdk_python._source_header import SOURCE_ENV_VAR, SOURCE_HEADER

MISSING = "MISSING"


class TestMissingWraptDependency:
    def test_import_error_message_mentions_extra(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "wrapt":
                raise ImportError("No module named 'wrapt'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "wrapt", raising=False)
        monkeypatch.delitem(
            sys.modules, "visionai_sdk_python.instrumentation", raising=False
        )

        with pytest.raises(
            ImportError, match="visionai-sdk-python\\[instrumentation\\]"
        ):
            import visionai_sdk_python.instrumentation  # noqa: F401


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/redirect"):
            query = urllib.parse.urlparse(self.path).query
            target = urllib.parse.parse_qs(query)["to"][0]
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        body = (self.headers.get(SOURCE_HEADER) or MISSING).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET

    def log_message(self, *args):
        pass


def _redirect(base: str, target: str) -> str:
    return f"{base}redirect?to={urllib.parse.quote(target, safe='')}"


@pytest.fixture(scope="module")
def _server():
    server = socketserver.TCPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def url(_server):
    return f"http://127.0.0.1:{_server.server_address[1]}/"


@pytest.fixture
def url_localhost(_server):
    """The same server, reached through a different hostname string.

    Lets destination-scoping tests exercise "this host is not on the allowlist"
    without standing up a second server: 127.0.0.1 and localhost both reach it,
    but only one of them can match a given allowlist pattern at a time.
    """
    return f"http://localhost:{_server.server_address[1]}/"


@pytest.fixture
def instrumented(monkeypatch):
    """Instrument with a known source and an allowlist covering the test server,
    and always unwrap afterwards."""
    monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
    instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
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

    def test_send_does_not_mutate_the_callers_prepared_request(
        self, url, url_localhost, instrumented
    ):
        """Regression: injection used to mutate request.headers in place. That
        left our value sitting on the caller's own PreparedRequest after
        send() returned, so reusing that same object for a second, unrelated
        send() (a legitimate requests pattern -- mutate .url, send again) saw
        our earlier value and mistook it for something the caller had set,
        keeping it even once the destination was no longer allowed."""
        session = requests.Session()
        prepared = session.prepare_request(requests.Request("GET", url))

        assert session.send(prepared).text == "stream-agent"
        assert SOURCE_HEADER not in prepared.headers  # caller's object untouched

        prepared.url = url_localhost  # not on the allowlist
        assert session.send(prepared).text == MISSING

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
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            async with aiohttp.ClientSession() as session:
                assert isinstance(session, aiohttp.ClientSession)
                assert type(session) is aiohttp.ClientSession

    async def test_preserves_caller_duplicate_headers(self, url, instrumented):
        """Injecting as a pair list must not collapse a caller's duplicate headers.

        Injection happens per-request (ClientRequest.__init__), not on the
        session's own default headers anymore, so this checks the request that
        actually reached the wire rather than session.headers."""
        from multidict import CIMultiDict

        headers = CIMultiDict([("X-Foo", "1"), ("X-Foo", "2")])
        async with aiohttp.ClientSession(headers=headers) as session:
            assert len(session.headers.getall("X-Foo")) == 2
            response = await session.get(url)
            assert await response.text() == "stream-agent"

    async def test_preserves_caller_headers_from_iterator(self, url, instrumented):
        """Inspecting a one-shot iterator must not consume its headers."""
        headers = iter([("X-Custom", "hello"), ("X-Trace", "abc123")])

        async with aiohttp.ClientSession(headers=headers) as session:
            assert session.headers["X-Custom"] == "hello"
            assert session.headers["X-Trace"] == "abc123"
            response = await session.get(url)
            assert await response.text() == "stream-agent"

    async def test_iterator_supplied_source_header_wins(self, url, instrumented):
        """A consumed iterator must be replaced even when no injection is needed."""
        headers = iter([("X-Custom", "hello"), (SOURCE_HEADER, "caller")])

        async with aiohttp.ClientSession(headers=headers) as session:
            assert session.headers["X-Custom"] == "hello"
            response = await session.get(url)
            assert await response.text() == "caller"


class TestNoOpByDefault:
    def test_no_env_var_means_no_header(self, url, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            assert requests.Session().get(url).text == MISSING
        finally:
            instrumentation.uninstrument()

    def test_uninstrument_restores_original_behavior(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        assert requests.Session().get(url).text == "stream-agent"

        instrumentation.uninstrument()

        assert requests.Session().get(url).text == MISSING

    def test_instrument_is_idempotent(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

        assert requests.Session().get(url).text == MISSING

    def test_a_later_call_still_updates_the_allowlist(self, url, monkeypatch):
        """The method wrapping only happens once, but the allowlist itself must
        not get stuck on whatever the first call passed -- a service that calls
        instrument() more than once (e.g. from two separate init helpers) should
        not have its later, intended allowlist silently ignored."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            assert requests.Session().get(url).text == MISSING

            instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_a_later_bare_call_does_not_reset_the_allowlist(self, url, monkeypatch):
        """Regression: allowed_destination_hosts used to be unconditionally
        reset to the default on every call, so a second, unrelated instrument()
        call with no arguments (e.g. from another init path, a library, a test
        fixture) silently wiped out a custom allowlist with no warning."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            assert requests.Session().get(url).text == "stream-agent"

            instrumentation.instrument()  # no args -- must be a no-op for scoping
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()


class TestMalformedEnvVar:
    """A bad env var must not be able to break the caller's traffic."""

    def test_trailing_newline_is_stripped(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent\n")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_unusable_value_is_dropped_and_request_still_succeeds(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream\nagent")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            with pytest.warns(RuntimeWarning):
                assert requests.Session().get(url).text == MISSING
        finally:
            instrumentation.uninstrument()


class TestStartupWarnsIfSourceEnvVarUnset:
    """Regression: allowed_destination_hosts tracks whether a *destination*
    matched, not whether a header was actually sent. A correct allowlist
    paired with an unset VISIONAI_SERVICE_SOURCE (e.g. left out of Helm
    values) means every call still reaches an allowed destination -- so
    nothing ever looks unmatched to TestLifetimeNeverMatchedWarning's
    tracking, even though no header goes out at all. instrument() checks
    the env var directly instead of relying on that tracking to catch it."""

    def test_warns_when_source_env_var_is_unset(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        try:
            with pytest.warns(RuntimeWarning, match=SOURCE_ENV_VAR):
                instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        finally:
            instrumentation.uninstrument()

    def test_no_warning_when_source_env_var_is_set(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        finally:
            instrumentation.uninstrument()

    def test_no_unset_warning_for_a_set_but_malformed_value(self, monkeypatch):
        """Regression: _source_value() returns None both when the env var is
        truly absent and when it's set but fails validation, so checking
        that return value directly would mislabel a malformed-but-present
        value as "unset". The malformed case already gets its own warning
        from _validate() -- this one must not also fire for it."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream\nagent")  # set, but invalid
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
            assert not any(SOURCE_ENV_VAR in str(w.message) for w in caught)
        finally:
            instrumentation.uninstrument()

    def test_correct_allowlist_does_not_mask_the_missing_env_var(
        self, url, monkeypatch
    ):
        """The exact repro from review: allowed_destination_hosts is
        correct (it matches the real destination), so every dispatch
        "matches" and TestLifetimeNeverMatchedWarning's tracking alone
        would stay silent forever -- this warning is what actually
        surfaces the missing env var."""
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        try:
            with pytest.warns(RuntimeWarning, match=SOURCE_ENV_VAR):
                instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
            assert requests.Session().get(url).text == MISSING
            assert instrumentation._ever_matched_destination is True
        finally:
            instrumentation.uninstrument()


class TestMissingPatchTargetWarnsInsteadOfSilentlyDoingNothing:
    """Regression: instrument() caught ImportError and AttributeError alike.
    ImportError (library not installed) is an expected skip, but AttributeError
    means a patch target we assumed exists on a private upstream API is gone
    (e.g. a minor version renamed/removed it) -- that used to fail exactly the
    same way, silently, with no signal that attribution had quietly stopped
    working for that library."""

    def test_missing_target_warns_but_other_targets_still_get_patched(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        original = httpx.Client._send_single_request
        del httpx.Client._send_single_request
        try:
            with pytest.warns(
                RuntimeWarning, match="httpx.Client._send_single_request"
            ):
                instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])

            # requests wasn't affected by httpx's missing target -- still patched.
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()
            httpx.Client._send_single_request = original


class TestDestinationScoping:
    """allowed_destination_hosts: only allowlisted destinations get the header,
    fail-closed, no unrestricted mode."""

    def test_first_call_without_a_list_raises(self, monkeypatch):
        """No default, no "match everything" -- a service must decide. Guessing
        one that doesn't match a service's real hosts would silently inject
        nothing, indistinguishable from attribution never being wired up."""
        with pytest.raises(ValueError, match="allowed_destination_hosts"):
            instrumentation.instrument()

    def test_empty_list_raises(self, monkeypatch):
        """Regression: an empty list passed `is not None`, so it silently
        produced an allowlist that matches nothing -- indistinguishable from
        the allowlist simply not being configured, but with no warning at
        all, unlike the omitted-argument case above."""
        with pytest.raises(ValueError, match="allowed_destination_hosts"):
            instrumentation.instrument(allowed_destination_hosts=[])

    def test_blank_strings_only_raises(self, monkeypatch):
        """Regression: [""] (e.g. from os.environ.get(...).split(",") on an
        unset/empty env var) also `is not None` and also matches nothing."""
        with pytest.raises(ValueError, match="allowed_destination_hosts"):
            instrumentation.instrument(allowed_destination_hosts=["", "  "])

    def test_explicit_allowlist_permits_a_named_host(self, url, instrumented):
        assert requests.Session().get(url).text == "stream-agent"

    def test_destination_outside_the_allowlist_gets_no_header(
        self, url_localhost, instrumented
    ):
        assert requests.Session().get(url_localhost).text == MISSING

    def test_httpx_destination_outside_the_allowlist_gets_no_header(
        self, url_localhost, instrumented
    ):
        with httpx.Client() as client:
            assert client.get(url_localhost).text == MISSING

    async def test_aiohttp_destination_outside_the_allowlist_gets_no_header(
        self, url_localhost, instrumented
    ):
        async with aiohttp.ClientSession() as session:
            response = await session.get(url_localhost)
            assert await response.text() == MISSING

    def test_caller_supplied_value_is_preserved_even_off_the_allowlist(
        self, url_localhost, instrumented
    ):
        response = requests.Session().get(
            url_localhost, headers={SOURCE_HEADER: "caller"}
        )
        assert response.text == "caller"

    def test_escape_hatch_reaches_a_disallowed_destination(
        self, url_localhost, instrumented
    ):
        """source_headers() is documented as the way to deliberately send the
        header to a destination outside the allowlist. Its value equals
        exactly what we'd inject ourselves, so this is the case that must not
        be confused -- by value alone -- with our own value and stripped."""
        response = requests.Session().get(
            url_localhost, headers={**instrumentation.source_headers()}
        )
        assert response.text == "stream-agent"

    async def test_httpx_escape_hatch_reaches_a_disallowed_destination(
        self, url_localhost, instrumented
    ):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url_localhost, headers={**instrumentation.source_headers()}
            )
        assert response.text == "stream-agent"

    async def test_aiohttp_escape_hatch_reaches_a_disallowed_destination(
        self, url_localhost, instrumented
    ):
        async with aiohttp.ClientSession() as session:
            response = await session.get(
                url_localhost, headers={**instrumentation.source_headers()}
            )
            assert await response.text() == "stream-agent"

    def test_wildcard_pattern_matches_the_parsed_host(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.*"])
        try:
            assert requests.Session().get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_hostname_match_is_case_insensitive(self, url_localhost, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["LOCALHOST"])
        try:
            assert requests.Session().get(url_localhost).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_matching_is_against_the_parsed_host_not_the_whole_url(
        self, url, monkeypatch
    ):
        """A pattern that would match if we naively searched the whole URL
        string must not match just because it appears in the path -- only the
        parsed hostname is compared."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["*evil*"])
        try:
            response = requests.Session().get(url + "evil-looking-path")
            assert response.text == MISSING
        finally:
            instrumentation.uninstrument()


class TestLifetimeNeverMatchedWarning:
    """A syntactically fine allowlist that never matches anything the service
    actually calls looks identical, from the outside, to attribution simply
    having no data for some other reason -- _warn_if_never_matched() is the
    signal that tells them apart. Registered via atexit for real usage; called
    directly here since atexit hooks aren't reliably observable from within
    the same test process."""

    def test_warns_when_nothing_ever_matched(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            requests.Session().get(url)  # dispatched, but doesn't match
            with pytest.warns(RuntimeWarning, match="never matched"):
                instrumentation._warn_if_never_matched()
        finally:
            instrumentation.uninstrument()

    def test_warns_even_without_source_env_var_configured(self, url, monkeypatch):
        """Regression: _destination_allowed() (which feeds _note_dispatch())
        was only called when _effective_source() was truthy, so with
        VISIONAI_SERVICE_SOURCE unset -- itself a likely misconfiguration --
        no dispatch was ever recorded and this warning could never fire,
        leaving the allowlist-typo case silently indistinguishable from it."""
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            requests.Session().get(url)  # dispatched, but doesn't match
            with pytest.warns(RuntimeWarning, match="never matched"):
                instrumentation._warn_if_never_matched()
        finally:
            instrumentation.uninstrument()

    def test_early_warning_fires_on_the_next_dispatch_once_the_threshold_elapses(
        self, url, monkeypatch
    ):
        """Regression: atexit does not run on a bare SIGTERM (the usual
        container shutdown signal), so a killed container never saw this
        warning at all. Once _EARLY_WARN_SECONDS has passed since
        instrument() with still no match, the very next unmatched dispatch
        warns immediately -- no atexit, no process exit required."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            monkeypatch.setattr(
                instrumentation, "_instrumented_at", time.monotonic() - 31
            )
            with pytest.warns(RuntimeWarning, match="never matched"):
                requests.Session().get(url)  # dispatched, doesn't match
        finally:
            instrumentation.uninstrument()

    def test_no_early_warning_before_the_threshold_elapses(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                requests.Session().get(url)  # dispatched, doesn't match, too soon
        finally:
            instrumentation.uninstrument()

    def test_early_warning_does_not_fire_twice_at_shutdown(self, url, monkeypatch):
        """The early check and the atexit-registered check share the same
        _warn_if_never_matched(), guarded so the same misconfiguration isn't
        reported twice."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            monkeypatch.setattr(
                instrumentation, "_instrumented_at", time.monotonic() - 31
            )
            with pytest.warns(RuntimeWarning, match="never matched"):
                requests.Session().get(url)  # early warning fires here

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                instrumentation._warn_if_never_matched()  # shutdown: must not repeat
        finally:
            instrumentation.uninstrument()

    def test_no_warning_once_something_matched(self, url, instrumented):
        requests.Session().get(
            url
        )  # matches -- allowed_destination_hosts=["127.0.0.1"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            instrumentation._warn_if_never_matched()  # must not raise/warn

    def test_no_warning_if_nothing_was_ever_dispatched(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["example.invalid"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                instrumentation._warn_if_never_matched()  # no requests made at all
        finally:
            instrumentation.uninstrument()


class TestRedirectReChecksDestinationPerHop:
    def test_requests_allowed_to_allowed_keeps_the_header(self, url, instrumented):
        response = requests.Session().get(_redirect(url, url))
        assert response.text == "stream-agent"

    def test_requests_allowed_to_disallowed_strips_the_header(
        self, url, url_localhost, instrumented
    ):
        response = requests.Session().get(_redirect(url, url_localhost))
        assert response.text == MISSING

    def test_httpx_allowed_to_disallowed_strips_the_header(
        self, url, url_localhost, instrumented
    ):
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(_redirect(url, url_localhost))
        assert response.text == MISSING

    def test_httpx_allowed_to_allowed_keeps_the_header(self, url, instrumented):
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(_redirect(url, url))
        assert response.text == "stream-agent"

    async def test_aiohttp_allowed_to_disallowed_strips_the_header(
        self, url, url_localhost, instrumented
    ):
        async with aiohttp.ClientSession() as session:
            response = await session.get(_redirect(url, url_localhost))
            assert await response.text() == MISSING

    async def test_aiohttp_allowed_to_allowed_keeps_the_header(self, url, instrumented):
        async with aiohttp.ClientSession() as session:
            response = await session.get(_redirect(url, url))
            assert await response.text() == "stream-agent"

    def test_escape_hatch_value_survives_a_redirect_to_a_disallowed_host(
        self, url, url_localhost, instrumented
    ):
        """The escape hatch is caller intent, honored for the whole chain --
        unlike our own injected value, which a later disallowed hop strips."""
        response = requests.Session().get(
            _redirect(url, url_localhost),
            headers={**instrumentation.source_headers()},
        )
        assert response.text == "stream-agent"


class TestRequestsNestedCallDoesNotLeakDispatchState:
    """Regression: a response hook fires from inside Session.send() -- still on
    the same call stack, before the outer send() returns -- and can make its
    own, wholly independent request. That nested request used to incorrectly
    inherit the outer chain's "we_injected" state (both share the same
    contextvar-held dict), which could strip a caller-supplied header on the
    nested request that had nothing to do with the outer one. State is now
    keyed by id(request.hooks), which PreparedRequest.copy() carries by
    reference across a real redirect chain but which a fresh, unrelated
    request never shares.
    """

    def test_hook_triggered_request_keeps_its_own_explicit_header(
        self, url, url_localhost, instrumented
    ):
        nested_result = {}

        def hook(response, *args, **kwargs):
            nested_result["text"] = requests.get(
                url_localhost, headers={"X-Request-Source": "audit-tool"}
            ).text

        response = requests.Session().get(url, hooks={"response": [hook]})

        assert response.text == "stream-agent"  # outer: our own injection
        assert (
            nested_result["text"] == "audit-tool"
        )  # inner: caller's own value, untouched

    """Injection now patches a request-dispatch method looked up on the class at
    call time (Session.send / Client._send_single_request /
    ClientRequest.__init__), not __init__. Unlike the old construction-time
    patch, a client built before instrument() runs is not a gap: it looks up the
    same (now-patched) method on every call it makes afterwards, so there is no
    "instrument() ran too late" failure mode left for these three libraries."""

    def test_requests_session_built_before_instrument_is_still_instrumented(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.uninstrument()

        session = requests.Session()  # built before instrument() runs
        try:
            instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
            assert session.get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()

    def test_httpx_client_built_before_instrument_is_still_instrumented(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.uninstrument()

        client = httpx.Client()  # built before instrument() runs
        try:
            instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
            assert client.get(url).text == "stream-agent"
        finally:
            instrumentation.uninstrument()
            client.close()

    async def test_aiohttp_session_built_before_instrument_is_still_instrumented(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.uninstrument()

        async with aiohttp.ClientSession() as session:  # built before instrument()
            try:
                instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
                response = await session.get(url)
                assert await response.text() == "stream-agent"
            finally:
                instrumentation.uninstrument()


class TestSdkClientCarriesHeaderNatively:
    """The SDK's own clients must attribute themselves without instrument().

    Injection happens per-request (in ``_request``, via
    ``merge_request_attribution``), not once at client construction -- a
    static construction-time default header looked, to instrument()'s
    per-request scoping, indistinguishable from a caller-supplied value,
    which silently defeated both destination scoping and origin forwarding
    for the SDK's own clients whenever instrument() was also active. See
    ``test_cooperates_with_instrument_destination_scoping`` below.
    """

    def test_sync_client_carries_header(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)

        assert client._request("GET", url).text == "stream-agent"

    async def test_async_client_carries_header(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import AsyncClient

        client = AsyncClient(auth_url=url, vlm_url=url)

        assert (await client._request("GET", url)).text == "stream-agent"

    def test_no_header_when_env_unset(self, url, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)

        assert client._request("GET", url).text == MISSING

    def test_caller_supplied_header_is_never_overridden(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)
        response = client._request(
            "GET", url, headers={"X-Request-Source": "caller-value"}
        )

        assert response.text == "caller-value"

    def test_built_before_scope_still_picks_up_a_later_current_origin(
        self, url, monkeypatch
    ):
        """Regression: construction-time injection meant a client built before
        entering a current_origin() scope never saw it. Per-request merging
        re-reads the origin on every call instead."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)  # built outside any scope

        with instrumentation.current_origin("observ-pod-1"):
            assert client._request("GET", url).text == "observ-pod-1"

    def test_caller_supplied_pairs_list_header_is_preserved(self, url, monkeypatch):
        """Regression: merge_request_attribution() assumed headers was a
        mapping and iterated it directly, so a caller passing a list of
        (key, value) pairs -- valid httpx usage -- raised AttributeError on
        ``key.lower()`` instead of being merged with."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)
        response = client._request(
            "GET", url, headers=[("X-Request-Source", "caller-value")]
        )

        assert response.text == "caller-value"

    def test_pairs_list_header_still_gets_the_header_injected(self, url, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)
        response = client._request("GET", url, headers=[("X-Other", "value")])

        assert response.text == "stream-agent"

    def test_cooperates_with_instrument_destination_scoping(
        self, url, url_localhost, monkeypatch
    ):
        """Regression: a static default header was indistinguishable from a
        caller-supplied one, so instrument()'s per-request scoping treated it
        as "not ours" and left it alone unconditionally -- sending it to any
        destination the client talked to, allowlisted or not. Tagging the
        injection via the same INJECTED_EXTENSION_KEY marker instrument() uses
        makes the client's own header subject to the same scoping."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            from visionai_sdk_python import Client

            allowed_client = Client(auth_url=url, vlm_url=url)
            assert allowed_client._request("GET", url).text == "stream-agent"

            disallowed_client = Client(auth_url=url_localhost, vlm_url=url_localhost)
            assert disallowed_client._request("GET", url_localhost).text == MISSING
        finally:
            instrumentation.uninstrument()


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

    def test_extracts_from_raw_asgi_byte_pairs(self):
        """Regression: scope["headers"] (ASGI's raw wire-level representation,
        used by middleware that doesn't go through a framework's already-decoded
        headers object) is list[tuple[bytes, bytes]] -- bytes never compares
        equal to the str SOURCE_HEADER regardless of content, which silently
        dropped the inherited origin instead of erroring."""
        headers = [
            (b"content-type", b"application/json"),
            (b"x-request-source", b"observ-pod-1"),
        ]

        result = instrumentation.origin_from_headers(headers)

        assert result == "observ-pod-1"
        assert isinstance(result, str)  # not bytes -- current_origin() requires str

    def test_returned_value_feeds_current_origin_without_crashing(self):
        """The value from origin_from_headers() must be usable directly as
        current_origin()'s argument -- decoding only the key for comparison and
        forgetting the value would trade the silent-drop bug for a TypeError
        here, since _validate() matches against a str-only regex."""
        headers = [(b"x-request-source", b"observ-pod-1")]

        origin = instrumentation.origin_from_headers(headers)
        with instrumentation.current_origin(origin):
            assert instrumentation.get_current_origin() == "observ-pod-1"


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
        """Client built before the scope, matching real startup-then-reuse
        usage -- building it inside the scope would only exercise "construction
        happened to run while the scope was active," not real per-request
        inheritance (see TestSdkClientCarriesHeaderNatively for that
        regression covered directly)."""
        from visionai_sdk_python import Client

        client = Client(auth_url=url, vlm_url=url)

        with instrumentation.current_origin("observ-pod-1"):
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

    def test_shared_client_inherits_a_later_scope_without_manual_merging(
        self, url, instrumented
    ):
        """Injection happens at send-time rather than construction-time, so a
        client built once outside any scope still picks up a current_origin()
        scope entered later, on every call made through it in that scope -- no
        manual source_headers() merge needed for this case anymore."""
        shared = requests.Session()  # built outside any scope

        with instrumentation.current_origin("observ-pod-1"):
            assert shared.get(url).text == "observ-pod-1"

        assert shared.get(url).text == "stream-agent"

    def test_per_call_header_overrides_a_shared_clients_default(
        self, url, instrumented
    ):
        """source_headers() remains available as an escape hatch, e.g. for
        sending the header to a destination outside allowed_destination_hosts on
        purpose. Per-call headers already override a client's defaults."""
        shared = requests.Session()

        with instrumentation.current_origin("observ-pod-1"):
            response = shared.get(url, headers={**instrumentation.source_headers()})

        assert response.text == "observ-pod-1"

    def test_shared_client_workaround_falls_back_to_own_identity_not_none(
        self, url, instrumented
    ):
        """larryyu1285's repro: outside any scope, get_current_origin() is None, and
        passing that raw None as a header value breaks both libraries -- requests
        silently drops the header (discarding the client's own correct default in
        the process) and httpx raises TypeError. source_headers() must be used
        instead, since it falls back to this service's own identity as a dict."""
        assert instrumentation.get_current_origin() is None

        shared = requests.Session()
        response = shared.get(url, headers={**instrumentation.source_headers()})

        assert response.text == "stream-agent"

    async def test_shared_httpx_client_workaround_does_not_crash(
        self, url, instrumented
    ):
        assert instrumentation.get_current_origin() is None

        async with httpx.AsyncClient() as shared:
            response = await shared.get(
                url, headers={**instrumentation.source_headers()}
            )

        assert response.text == "stream-agent"

    def test_raw_get_current_origin_as_a_header_value_is_no_longer_lost_for_requests(
        self, url, instrumented
    ):
        """This used to document a bug: requests treats a None-valued header as
        "remove this", which used to wipe out a default baked in at construction
        time. Now that injection happens at send-time instead, our own check
        re-adds the header regardless -- a side effect of the same redesign that
        fixed A-5's shared-client limitation, not something specifically
        targeted. source_headers() is still the documented, contract-guaranteed
        way to do this; httpx still hard-crashes on a raw None below, which this
        pattern remains a bad idea for."""
        assert instrumentation.get_current_origin() is None

        shared = requests.Session()
        response = shared.get(
            url, headers={"X-Request-Source": instrumentation.get_current_origin()}
        )
        assert response.text == "stream-agent"

        async def crashes():
            async with httpx.AsyncClient() as shared_h:
                return await shared_h.get(
                    url,
                    headers={"X-Request-Source": instrumentation.get_current_origin()},
                )

        import asyncio

        with pytest.raises(TypeError):
            asyncio.run(crashes())


class TestPodChainScenario:
    """The exact scenario from the review discussion: observ-pod-1 -> observ-pod-2 -> VLM,
    where pod-2's own request-handling code uses current_origin()."""

    def test_pod_2_forwards_pod_1s_identity_when_using_current_origin(
        self, url, monkeypatch
    ):
        monkeypatch.setenv(SOURCE_ENV_VAR, "observ-pod-2")
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
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
        instrumentation.instrument(allowed_destination_hosts=["127.0.0.1"])
        try:
            response = requests.Session().get(url)
        finally:
            instrumentation.uninstrument()

        assert response.text == "observ-pod-2"
