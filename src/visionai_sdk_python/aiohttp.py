"""Drop-in replacement for ``aiohttp`` — swap the import, ClientSession auto-carries X-Request-Source.

    from visionai_sdk_python import aiohttp   # was: import aiohttp

Requires the optional dependency: ``pip install visionai-sdk-python[aiohttp]``.

``ClientSession`` only overrides ``__init__`` to set a default header —
aiohttp merges default + per-call headers itself, per-call wins. This
triggers aiohttp's subclassing ``DeprecationWarning``; accepted trade-off
so ``isinstance(session, ...ClientSession)`` still holds after the import swap.
"""

from typing import Any

try:
    import aiohttp as _aiohttp
except ImportError as e:
    raise ImportError(
        "visionai_sdk_python.aiohttp requires the 'aiohttp' package. "
        "Install it with: pip install visionai-sdk-python[aiohttp]"
    ) from e

from ._source_header import merge_source_headers


class ClientSession(_aiohttp.ClientSession):
    """``aiohttp.ClientSession`` subclass with ``X-Request-Source`` as a default header."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
        super().__init__(*args, **kwargs)


def request(method: str, url: Any, *args: Any, **kwargs: Any) -> Any:
    """Wrap ``aiohttp.request()`` (module-level one-shot call, no ClientSession involved)."""
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _aiohttp.request(method, url, *args, **kwargs)


def __getattr__(name: str) -> Any:
    """Delegate anything not defined here to the real ``aiohttp`` (e.g. exceptions, ClientTimeout)."""
    return getattr(_aiohttp, name)
