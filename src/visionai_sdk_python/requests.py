"""Drop-in replacement for the ``requests`` library.

Same interface as ``requests`` — swap the import and every outbound call
automatically carries ``X-Request-Source`` (read from the
``VISIONAI_SERVICE_SOURCE`` env var). This module is a pure passthrough:
it does not catch, translate, or wrap anything ``requests`` raises or
returns, so existing ``try``/``except`` and ``isinstance`` checks around
``requests`` exceptions keep working unchanged.

    # before
    import requests
    requests.post(url, json=payload, headers=h)

    # after -- only this line changes
    from visionai_sdk_python import requests
    requests.post(url, json=payload, headers=h)

Requires the optional ``requests`` dependency:
``pip install visionai-sdk-python[requests]``.
"""

from typing import Any

try:
    import requests as _requests
except ImportError as e:
    raise ImportError(
        "visionai_sdk_python.requests requires the 'requests' package. "
        "Install it with: pip install visionai-sdk-python[requests]"
    ) from e

from ._source_header import merge_source_headers


class Session(_requests.Session):
    """``requests.Session`` subclass that auto-injects ``X-Request-Source``.

    Everything else (mount, adapters, cookies, auth, hooks, ...) is the
    real ``requests.Session`` behavior, unmodified.
    """

    def request(
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> _requests.Response:
        kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
        return super().request(method, url, *args, **kwargs)


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
