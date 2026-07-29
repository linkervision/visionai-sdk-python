"""Drop-in replacement for ``requests`` — swap the import, requests auto-carry X-Request-Source.

    from visionai_sdk_python import requests   # was: import requests

Pure passthrough: no exception translation, so existing try/except and
isinstance checks around requests exceptions keep working. Requires the
optional dependency: ``pip install visionai-sdk-python[requests]``.
"""

from typing import Any

try:
    import requests as _requests
except ImportError as e:
    raise ImportError(
        "visionai_sdk_python.requests requires the 'requests' package. "
        "Install it with: pip install visionai-sdk-python[requests]"
    ) from e

from ._source_header import inject_source_header, merge_source_headers


class Session(_requests.Session):
    """``requests.Session`` subclass that auto-injects ``X-Request-Source``.

    Overrides ``send()``, not ``request()`` — by then session + per-call
    headers are already merged, and it also covers manual ``send(prepared)`` calls.
    """

    def send(
        self, request: _requests.PreparedRequest, **kwargs: Any
    ) -> _requests.Response:
        inject_source_header(request.headers)
        return super().send(request, **kwargs)


def session() -> Session:
    """Return this module's ``Session`` (overrides the real lib's lowercase factory)."""
    return Session()


def request(method: str, url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.request(method, url, *args, **kwargs)


def get(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.get(url, *args, **kwargs)


def options(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.options(url, *args, **kwargs)


def head(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.head(url, *args, **kwargs)


def post(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.post(url, *args, **kwargs)


def put(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.put(url, *args, **kwargs)


def patch(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.patch(url, *args, **kwargs)


def delete(url: str, *args: Any, **kwargs: Any) -> _requests.Response:
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _requests.delete(url, *args, **kwargs)


def __getattr__(name: str) -> Any:
    """Delegate anything not defined here to the real ``requests`` (e.g. exceptions, Response)."""
    return getattr(_requests, name)
