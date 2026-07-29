"""Drop-in replacement for ``httpx`` — swap the import, requests auto-carry X-Request-Source.

    from visionai_sdk_python import httpx   # was: import httpx

Pure passthrough, no exception translation. ``httpx`` is already a core
dependency, no optional extra needed. Shares its name with the real
``httpx`` package deliberately — absolute imports mean ``import httpx``
below still resolves to the real package.

``Client``/``AsyncClient`` override ``send()``, the single choke point
after headers are already merged (also covers manual ``send(request)`` calls).
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
    """Delegate anything not defined here to the real ``httpx`` (e.g. exceptions, Response)."""
    return getattr(_httpx, name)
