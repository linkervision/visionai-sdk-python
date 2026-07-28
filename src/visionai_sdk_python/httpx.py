"""Drop-in replacement for the ``httpx`` library.

Same interface as ``httpx`` -- swap the import and every outbound call
automatically carries ``X-Request-Source`` (read from the
``VISIONAI_SERVICE_SOURCE`` env var). This module is a pure passthrough:
it does not catch, translate, or wrap anything ``httpx`` raises or
returns, so existing ``try``/``except`` and ``isinstance`` checks around
``httpx`` exceptions keep working unchanged.

    # before
    import httpx
    httpx.post(url, json=payload, headers=h)

    # after -- only this line changes
    from visionai_sdk_python import httpx
    httpx.post(url, json=payload, headers=h)

``httpx`` is already a core dependency of this SDK, so no optional extra
is required here (unlike the ``requests``/``aiohttp`` shims).

Note: this module intentionally shares its name with the third-party
``httpx`` package. Python 3 imports are absolute by default, so
``import httpx`` below resolves to the real third-party package, not
this module -- a deliberate naming choice, not a mistake, though it can
confuse IDEs/type-checkers that don't model that distinction.

``Client``/``AsyncClient`` override ``send()`` rather than
``build_request()``/``request()``: ``send()`` is the single choke point
the real ``httpx.Client``/``AsyncClient`` use to actually transmit every
request, reached by ``.request()``, ``.get()``, ``.post()``, ``.stream()``
-- and also by a caller who builds a ``Request`` manually (or via
``build_request()``) and calls ``client.send(request)`` directly, a
supported public workflow that ``build_request()`` alone would miss
entirely. By the time a request reaches ``send()``, ``httpx`` has already
merged the client's own default headers with any per-call headers, so a
single presence check there covers both cases with one override.
"""

from typing import Any

import httpx as _httpx

from ._source_header import inject_source_header, merge_source_headers


class Client(_httpx.Client):
    """``httpx.Client`` subclass that auto-injects ``X-Request-Source``."""

    def send(self, request: _httpx.Request, **kwargs: Any) -> _httpx.Response:
        inject_source_header(request.headers)
        return super().send(request, **kwargs)


class AsyncClient(_httpx.AsyncClient):
    """``httpx.AsyncClient`` subclass that auto-injects ``X-Request-Source``."""

    async def send(self, request: _httpx.Request, **kwargs: Any) -> _httpx.Response:
        inject_source_header(request.headers)
        return await super().send(request, **kwargs)


def request(method: str, url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.request(method, url, *args, **kwargs)


def get(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.get(url, *args, **kwargs)


def options(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.options(url, *args, **kwargs)


def head(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.head(url, *args, **kwargs)


def post(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.post(url, *args, **kwargs)


def put(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.put(url, *args, **kwargs)


def patch(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.patch(url, *args, **kwargs)


def delete(url: str, *args: Any, **kwargs: Any) -> _httpx.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.delete(url, *args, **kwargs)


def stream(method: str, url: str, *args: Any, **kwargs: Any) -> Any:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _httpx.stream(method, url, *args, **kwargs)


def __getattr__(name: str) -> Any:
    """Delegate anything this module doesn't define itself to real ``httpx``.

    Without this, code that references ``httpx.ConnectError``, ``httpx.Timeout``,
    ``httpx.Response``, ``httpx.HTTPError``, etc. through this shim's namespace
    breaks with ``AttributeError`` after the documented one-line import swap.
    """
    return getattr(_httpx, name)
