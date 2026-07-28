import httpx as real_httpx
import pytest

from visionai_sdk_python import httpx as shim
from visionai_sdk_python._source_header import SOURCE_ENV_VAR, SOURCE_HEADER


def _capturing_transport(captured: dict) -> real_httpx.MockTransport:
    def handler(request: real_httpx.Request) -> real_httpx.Response:
        captured["headers"] = dict(request.headers)
        return real_httpx.Response(200, request=request)

    return real_httpx.MockTransport(handler)


def _raising_transport(exc: Exception) -> real_httpx.MockTransport:
    def handler(request: real_httpx.Request) -> real_httpx.Response:
        raise exc

    return real_httpx.MockTransport(handler)


class TestClientHeaderInjection:
    def test_get_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        client = shim.Client(transport=_capturing_transport(captured))
        client.get("http://example.test/path", headers={"Authorization": "Bearer x"})

        assert captured["headers"]["x-request-source"] == "stream-agent"
        assert captured["headers"]["authorization"] == "Bearer x"

    def test_stream_injects_header(self, monkeypatch):
        """.stream() bypasses .request() entirely; this proves the
        build_request() override still covers it."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        client = shim.Client(transport=_capturing_transport(captured))
        with client.stream("GET", "http://example.test/path") as response:
            response.read()

        assert captured["headers"]["x-request-source"] == "stream-agent"

    def test_respects_caller_supplied_value(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        client = shim.Client(transport=_capturing_transport(captured))
        client.get(
            "http://example.test/path", headers={"X-Request-Source": "web-server"}
        )

        assert captured["headers"]["x-request-source"] == "web-server"

    def test_no_header_without_env(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        captured = {}

        client = shim.Client(transport=_capturing_transport(captured))
        client.get("http://example.test/path")

        assert SOURCE_HEADER.lower() not in captured["headers"]


class TestAsyncClientHeaderInjection:
    async def test_get_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "web-server")
        captured = {}

        client = shim.AsyncClient(transport=_capturing_transport(captured))
        await client.get("http://example.test/path")

        assert captured["headers"]["x-request-source"] == "web-server"

    async def test_stream_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "web-server")
        captured = {}

        client = shim.AsyncClient(transport=_capturing_transport(captured))
        async with client.stream("GET", "http://example.test/path") as response:
            await response.aread()

        assert captured["headers"]["x-request-source"] == "web-server"


class TestModuleLevelHeaderInjection:
    def test_get_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        monkeypatch.setattr(
            real_httpx.Client,
            "_transport_for_url",
            lambda self, url: _capturing_transport(captured),
        )
        shim.get("http://example.test/path")

        assert captured["headers"]["x-request-source"] == "stream-agent"

    def test_stream_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        monkeypatch.setattr(
            real_httpx.Client,
            "_transport_for_url",
            lambda self, url: _capturing_transport(captured),
        )
        with shim.stream("GET", "http://example.test/path") as response:
            response.read()

        assert captured["headers"]["x-request-source"] == "stream-agent"


class TestExceptionTransparency:
    def test_client_connection_error_propagates_untouched(self):
        client = shim.Client(
            transport=_raising_transport(real_httpx.ConnectError("Connection refused"))
        )
        with pytest.raises(real_httpx.ConnectError):
            client.get("http://example.test/path")

    async def test_async_client_connection_error_propagates_untouched(self):
        client = shim.AsyncClient(
            transport=_raising_transport(real_httpx.ConnectError("Connection refused"))
        )
        with pytest.raises(real_httpx.ConnectError):
            await client.get("http://example.test/path")

    def test_module_level_connection_error_propagates_untouched(self, monkeypatch):
        monkeypatch.setattr(
            real_httpx.Client,
            "_transport_for_url",
            lambda self, url: _raising_transport(
                real_httpx.ConnectError("Connection refused")
            ),
        )
        with pytest.raises(real_httpx.ConnectError):
            shim.get("http://example.test/path")
