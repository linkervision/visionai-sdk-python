"""Drop-in helper for the ``aiohttp`` library.

Same interface as ``aiohttp`` -- swap the import and the ``ClientSession``
it creates automatically carries ``X-Request-Source`` (read from the
``VISIONAI_SERVICE_SOURCE`` env var) on every request made through it.

    # before
    import aiohttp
    session = aiohttp.ClientSession()

    # after -- only this line changes
    from visionai_sdk_python import aiohttp
    session = aiohttp.ClientSession()

Requires the optional ``aiohttp`` dependency:
``pip install visionai-sdk-python[aiohttp]``.

Unlike the ``requests``/``httpx`` shims, this does not subclass anything
or intercept individual requests. ``aiohttp.ClientSession`` already
merges its own default headers with any headers passed per-call
(``session.get(url, headers=...)``), with the per-call value winning
for the same key (confirmed via ``ClientSession._prepare_headers``) --
so setting ``X-Request-Source`` once, here, at construction time, is
enough: a caller who passes their own ``X-Request-Source`` on a
specific request still gets that value, not the env default. The
object returned is the real ``aiohttp.ClientSession`` class, unmodified
-- ``isinstance`` checks against it, and everything about request
lifecycle/exceptions, keep working exactly as they do today.
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


def ClientSession(*args: Any, **kwargs: Any) -> _aiohttp.ClientSession:
    """Create a real ``aiohttp.ClientSession`` with ``X-Request-Source`` set as a default header."""
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _aiohttp.ClientSession(*args, **kwargs)
