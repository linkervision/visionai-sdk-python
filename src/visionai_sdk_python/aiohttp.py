"""Drop-in replacement for the ``aiohttp`` library.

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

``ClientSession`` is a subclass of ``aiohttp.ClientSession`` that only
overrides ``__init__`` to inject a default header -- it doesn't intercept
individual requests. ``aiohttp.ClientSession`` already merges its own
default headers with any headers passed per-call
(``session.get(url, headers=...)``), with the per-call value winning for
the same key (confirmed via ``ClientSession._prepare_headers``) -- so
setting ``X-Request-Source`` once, here, at construction time, is enough:
a caller who passes their own ``X-Request-Source`` on a specific request
still gets that value, not the env default.

Note: aiohttp's ``ClientSession.__init_subclass__`` emits a
``DeprecationWarning`` on every subclass of it, including this one --
aiohttp's maintainers discourage subclassing ``ClientSession`` in general
(see their own ``__init_subclass__`` implementation). This is a known,
accepted trade-off, not an oversight: the alternative (aiohttp's
``middlewares=`` constructor argument, which avoids the warning) requires
``ClientSession`` to be a plain factory function rather than a subclass --
and a factory function can never make
``isinstance(session, visionai_sdk_python.aiohttp.ClientSession)`` true,
since that requires ``ClientSession`` to actually be a type in ``session``'s
ancestry. Since real calling code doing that isinstance check (against the
same name they imported after the one-line swap) is exactly the failure
mode this shim exists to avoid, subclassing -- and accepting the
DeprecationWarning -- is the better trade-off here. If aiohttp ever turns
this into a hard error, revisit via ``middlewares=`` and accept the
isinstance gap instead (see git history for that implementation).
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
    """Wrap ``aiohttp.request()``, the module-level one-shot convenience function.

    Without this, ``aiohttp.request(...)`` (used without ever constructing
    a ``ClientSession``) would fall through ``__getattr__`` to the real
    function unchanged and never receive ``X-Request-Source`` -- unlike
    ``ClientSession``, it isn't a class we can subclass to inject a
    default, so it needs its own explicit wrapper here.
    """
    kwargs["headers"] = merge_source_headers(kwargs.get("headers"))
    return _aiohttp.request(method, url, *args, **kwargs)


def __getattr__(name: str) -> Any:
    """Delegate anything this module doesn't define itself to real ``aiohttp``.

    Without this, code that references ``aiohttp.ClientTimeout``,
    ``aiohttp.ClientConnectorError``, etc. through this shim's namespace
    breaks with ``AttributeError`` after the documented one-line import swap.
    """
    return getattr(_aiohttp, name)
