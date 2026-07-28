import importlib

import aiohttp as real_aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from visionai_sdk_python import aiohttp as shim
from visionai_sdk_python._source_header import SOURCE_ENV_VAR, SOURCE_HEADER


class TestClientSessionDefaultHeader:
    async def test_injects_from_env(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        async with shim.ClientSession() as session:
            assert session.headers[SOURCE_HEADER] == "stream-agent"

    async def test_no_header_without_env(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)

        async with shim.ClientSession() as session:
            assert SOURCE_HEADER not in session.headers

    async def test_preserves_caller_supplied_headers(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        async with shim.ClientSession(headers={"Authorization": "Bearer x"}) as session:
            assert session.headers["Authorization"] == "Bearer x"
            assert session.headers[SOURCE_HEADER] == "stream-agent"

    async def test_respects_caller_supplied_value_case_insensitively(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        async with shim.ClientSession(
            headers={"x-request-source": "web-server"}
        ) as session:
            assert session.headers["x-request-source"] == "web-server"


class TestPerRequestOverride:
    async def test_per_request_header_wins_over_session_default(self, monkeypatch):
        """aiohttp's own header-merge (session default + per-call, per-call
        wins) is what protects a caller's explicit per-request
        X-Request-Source -- our shim never touches individual requests."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        async def handler(request: web.Request) -> web.Response:
            captured["headers"] = dict(request.headers)
            return web.Response(text="ok")

        app = web.Application()
        app.router.add_get("/path", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            async with shim.ClientSession() as session:
                async with session.get(
                    server.make_url("/path"), headers={SOURCE_HEADER: "web-server"}
                ) as response:
                    await response.read()
        finally:
            await server.close()

        assert captured["headers"][SOURCE_HEADER] == "web-server"

    async def test_no_override_uses_session_default(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        async def handler(request: web.Request) -> web.Response:
            captured["headers"] = dict(request.headers)
            return web.Response(text="ok")

        app = web.Application()
        app.router.add_get("/path", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            async with shim.ClientSession() as session:
                async with session.get(server.make_url("/path")) as response:
                    await response.read()
        finally:
            await server.close()

        assert captured["headers"][SOURCE_HEADER] == "stream-agent"


class TestExceptionTransparency:
    async def test_connection_error_propagates_untouched(self):
        async with shim.ClientSession() as session:
            with pytest.raises(real_aiohttp.ClientConnectorError):
                async with session.get(
                    "http://127.0.0.1:1/",
                    timeout=real_aiohttp.ClientTimeout(total=2),
                ):
                    pass

    async def test_connection_error_catchable_via_shim_namespace(self):
        """Regression test: code that does `from visionai_sdk_python import
        aiohttp` and then `except aiohttp.ClientConnectorError:` must still
        work -- this failed with AttributeError before the shim re-exported
        aiohttp's public API."""
        async with shim.ClientSession() as session:
            with pytest.raises(shim.ClientConnectorError):
                async with session.get(
                    "http://127.0.0.1:1/",
                    timeout=shim.ClientTimeout(total=2),
                ):
                    pass


class TestReExportsUnderlyingLibrary:
    """Regression tests for attribute access through the shim's own
    namespace -- code referencing aiohttp.ClientTimeout,
    aiohttp.ClientConnectorError, etc. via the post-import-swap name must
    keep working."""

    def test_client_timeout_reexported(self):
        assert shim.ClientTimeout is real_aiohttp.ClientTimeout

    def test_client_connector_error_reexported(self):
        assert shim.ClientConnectorError is real_aiohttp.ClientConnectorError

    async def test_isinstance_against_shim_client_session(self):
        """Regression test for the exact bug Codex found: ClientSession
        must be a real type so isinstance(session, aiohttp.ClientSession)
        -- checked against the *shim's own* name, as real calling code
        would after the import swap -- doesn't raise TypeError."""
        async with shim.ClientSession() as session:
            assert isinstance(session, shim.ClientSession)
            assert isinstance(session, real_aiohttp.ClientSession)


class TestMissingDependency:
    def test_import_error_message_mentions_extra(self, monkeypatch):
        import builtins
        import sys

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "aiohttp":
                raise ImportError("No module named 'aiohttp'")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("visionai_sdk_python.aiohttp", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            with pytest.raises(ImportError, match=r"visionai-sdk-python\[aiohttp\]"):
                importlib.import_module("visionai_sdk_python.aiohttp")
        finally:
            sys.modules.pop("visionai_sdk_python.aiohttp", None)
            monkeypatch.setattr(builtins, "__import__", real_import)
            importlib.import_module("visionai_sdk_python.aiohttp")
