"""Source attribution for outbound HTTP, with no call-site changes.

Call ``instrument()`` once at service startup, before any HTTP client is
constructed::

    from visionai_sdk_python import instrumentation
    instrumentation.instrument()

Every ``requests.Session``, ``httpx.Client``/``AsyncClient`` and
``aiohttp.ClientSession`` built after that carries ``X-Request-Source``,
including ones built by third-party code we cannot edit, and the module-level
one-shots (``requests.get()``) that create a client internally.

The real classes are wrapped in place rather than subclassed or shadowed, so
``isinstance``, exception identity and the libraries' public APIs are untouched.
"""

from typing import Any

import wrapt

from ._source_header import SOURCE_HEADER, _source_value

_instrumented = False


def _already_set(headers: Any) -> bool:
    pairs = list(headers.items()) if hasattr(headers, "items") else list(headers or [])
    return any(key.lower() == SOURCE_HEADER.lower() for key, _ in pairs)


def _set_after_init(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """For clients exposing a mutable public ``.headers`` (requests, httpx)."""
    result = wrapped(*args, **kwargs)

    source = _source_value()
    if source and not _already_set(instance.headers):
        instance.headers[SOURCE_HEADER] = source

    return result


def _set_via_init_kwarg(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """For aiohttp, whose default headers can only be set through ``__init__``.

    Passing a list of pairs rather than a dict keeps any duplicate headers the
    caller supplied. This is also the one injection vehicle ``aioresponses``
    can see, which a ``TraceConfig``-based approach cannot offer.
    """
    source = _source_value()
    if source:
        headers = kwargs.get("headers")
        if not _already_set(headers):
            pairs = (
                list(headers.items())
                if hasattr(headers, "items")
                else list(headers or [])
            )
            kwargs["headers"] = [*pairs, (SOURCE_HEADER, source)]

    return wrapped(*args, **kwargs)


# (module, attribute path, wrapper). Missing modules are skipped: requests and
# aiohttp are optional extras, httpx is a core dependency.
_TARGETS = (
    ("requests", "Session.__init__", _set_after_init),
    ("httpx", "Client.__init__", _set_after_init),
    ("httpx", "AsyncClient.__init__", _set_after_init),
    ("aiohttp", "ClientSession.__init__", _set_via_init_kwarg),
)


def instrument() -> None:
    """Wrap the installed HTTP clients. Idempotent; safe to call once at startup."""
    global _instrumented
    if _instrumented:
        return

    for module, target, wrapper in _TARGETS:
        try:
            wrapt.wrap_function_wrapper(module, target, wrapper)
        except (ImportError, AttributeError):
            continue

    _instrumented = True


def uninstrument() -> None:
    """Restore the original clients. Mainly for test isolation."""
    global _instrumented
    if not _instrumented:
        return

    for module, target, _ in _TARGETS:
        try:
            obj = __import__(module)
            class_name, attr = target.split(".")
            cls = getattr(obj, class_name)
        except (ImportError, AttributeError):
            continue

        # vars() rather than getattr(): looking the attribute up normally would
        # run the descriptor protocol and hand back a *bound* wrapper, whose
        # __wrapped__ is not the plain function we need to put back. Duck-type on
        # __wrapped__ rather than isinstance(ObjectProxy) — under wrapt 2.x
        # FunctionWrapper is not an ObjectProxy instance.
        original = getattr(vars(cls).get(attr), "__wrapped__", None)
        if original is not None:
            setattr(cls, attr, original)

    _instrumented = False
