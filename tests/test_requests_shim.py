import builtins
import importlib
import sys

import pytest
import requests as real_requests
from requests.adapters import HTTPAdapter

from visionai_sdk_python import requests as shim
from visionai_sdk_python._source_header import (
    SOURCE_ENV_VAR,
    SOURCE_HEADER,
    merge_source_headers,
)


def _raise_connection_error(self, request, *args, **kwargs):
    raise real_requests.exceptions.ConnectionError("Connection refused")


class TestMergeSourceHeaders:
    def test_injects_from_env(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        assert merge_source_headers(None) == {SOURCE_HEADER: "stream-agent"}
        assert merge_source_headers({"Authorization": "Bearer x"}) == {
            "Authorization": "Bearer x",
            SOURCE_HEADER: "stream-agent",
        }

    def test_no_env_leaves_headers_untouched(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)

        assert merge_source_headers(None) == {}
        assert merge_source_headers({"Authorization": "Bearer x"}) == {
            "Authorization": "Bearer x"
        }

    def test_respects_caller_supplied_value_case_insensitively(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")

        result = merge_source_headers({"x-request-source": "web-server"})
        assert result == {"x-request-source": "web-server"}


class TestRequestsShimHeaderInjection:
    def test_module_level_get_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        def fake_send(self, request, *args, **kwargs):
            captured["headers"] = dict(request.headers)
            return real_requests.Response()

        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        shim.get("http://example.test/path", headers={"Authorization": "Bearer x"})

        assert captured["headers"][SOURCE_HEADER] == "stream-agent"
        assert captured["headers"]["Authorization"] == "Bearer x"

    def test_session_injects_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "web-server")
        captured = {}

        def fake_send(self, request, *args, **kwargs):
            captured["headers"] = dict(request.headers)
            return real_requests.Response()

        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        session = shim.Session()
        session.request("GET", "http://example.test/path")

        assert captured["headers"][SOURCE_HEADER] == "web-server"

    def test_session_convenience_methods_inject_header(self, monkeypatch):
        monkeypatch.setenv(SOURCE_ENV_VAR, "web-server")
        captured = {}

        def fake_send(self, request, *args, **kwargs):
            captured.setdefault("headers", []).append(dict(request.headers))
            return real_requests.Response()

        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        session = shim.Session()
        session.get("http://example.test/path")
        session.post("http://example.test/path", json={})

        assert all(h[SOURCE_HEADER] == "web-server" for h in captured["headers"])

    def test_no_header_without_env(self, monkeypatch):
        monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
        captured = {}

        def fake_send(self, request, *args, **kwargs):
            captured["headers"] = dict(request.headers)
            return real_requests.Response()

        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        shim.get("http://example.test/path")

        assert SOURCE_HEADER not in captured["headers"]

    def test_session_send_prepared_request_injects_header(self, monkeypatch):
        """Regression test: session.send(prepared_request) is a supported
        public requests workflow that bypasses request() entirely -- it
        must still get the header, since Session overrides send(), not
        request()."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        def fake_send(self, request, *args, **kwargs):
            captured["headers"] = dict(request.headers)
            return real_requests.Response()

        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        session = shim.Session()
        req = real_requests.Request("GET", "http://example.test/path")
        prepped = session.prepare_request(req)
        session.send(prepped)

        assert captured["headers"][SOURCE_HEADER] == "stream-agent"

    def test_session_factory_returns_shim_session(self, monkeypatch):
        """Regression test: requests.session() (the real library's own
        lowercase factory, still supported though deprecated) must return
        this module's Session, not fall through __getattr__ to a plain
        real requests.Session that never injects anything."""
        monkeypatch.setenv(SOURCE_ENV_VAR, "stream-agent")
        captured = {}

        def fake_send(self, request, *args, **kwargs):
            captured["headers"] = dict(request.headers)
            return real_requests.Response()

        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        session = shim.session()
        assert isinstance(session, shim.Session)
        session.get("http://example.test/path")

        assert captured["headers"][SOURCE_HEADER] == "stream-agent"


class TestExceptionTransparency:
    def test_module_level_connection_error_propagates_untouched(self, monkeypatch):
        monkeypatch.setattr(HTTPAdapter, "send", _raise_connection_error)

        with pytest.raises(real_requests.exceptions.ConnectionError):
            shim.get("http://example.test/path")

    def test_session_connection_error_propagates_untouched(self, monkeypatch):
        monkeypatch.setattr(HTTPAdapter, "send", _raise_connection_error)

        with pytest.raises(real_requests.exceptions.ConnectionError):
            shim.Session().request("GET", "http://example.test/path")

    def test_connection_error_catchable_via_shim_namespace(self, monkeypatch):
        """Regression test: code that does `from visionai_sdk_python import
        requests` and then `except requests.exceptions.ConnectionError:`
        must still work -- this failed with AttributeError before the shim
        re-exported requests' public API."""
        monkeypatch.setattr(HTTPAdapter, "send", _raise_connection_error)

        with pytest.raises(shim.exceptions.ConnectionError):
            shim.get("http://example.test/path")


class TestReExportsUnderlyingLibrary:
    """Regression tests for attribute access through the shim's own
    namespace -- code referencing requests.Response, requests.adapters.*,
    etc. via the post-import-swap name must keep working, not just code
    that imports the real `requests` package separately."""

    def test_exceptions_module_reexported(self):
        assert (
            shim.exceptions.ConnectionError is real_requests.exceptions.ConnectionError
        )

    def test_response_class_reexported(self):
        assert shim.Response is real_requests.Response

    def test_adapters_module_reexported(self):
        assert shim.adapters.HTTPAdapter is real_requests.adapters.HTTPAdapter


class TestMissingDependency:
    def test_import_error_message_mentions_extra(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("No module named 'requests'")
            return real_import(name, *args, **kwargs)

        sys.modules.pop("visionai_sdk_python.requests", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            with pytest.raises(ImportError, match=r"visionai-sdk-python\[requests\]"):
                importlib.import_module("visionai_sdk_python.requests")
        finally:
            sys.modules.pop("visionai_sdk_python.requests", None)
            monkeypatch.setattr(builtins, "__import__", real_import)
            importlib.import_module("visionai_sdk_python.requests")
