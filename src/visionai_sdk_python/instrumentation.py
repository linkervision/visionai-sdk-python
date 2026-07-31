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

**Known limitation — this stamps the local service's identity, it does not
propagate an inherited one.** ``instrument()`` always injects this process's own
``VISIONAI_SERVICE_SOURCE``; it never reads an inbound request's
``X-Request-Source`` and forwards it. That is correct as long as a service is
the true origin of whatever VLM call it makes. If service A calls service B
(both instrumented) and B then calls VLM on A's behalf, B's outbound call gets
*B's* identity, not A's — A's identity is silently lost at that hop, the same
problem ``visionai-vlm-scheduling-service`` solves for its Redis queue hop by
explicitly storing and re-attaching the header. Forwarding an inherited origin
requires request-scoped context (extract on inbound, inject on every outbound
call made while handling that request) — a different mechanism from this
module's process-lifetime environment variable, and out of scope here. See
"A-5" in the service-source-attribution plan. Today's known callers
(data-engine/observ/mirra) each call VLM independently rather than through each
other, so this gap does not currently affect correctness — but the first
service-to-service chain between two SDK-instrumented services will need this
feature before it can be attributed correctly.
"""

import sys
import warnings
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


class LateInstrumentationWarning(RuntimeWarning):
    """Raised when instrument() runs after an HTTP library was already imported.

    Modelled on ``gevent.monkey``'s ``MonkeyPatchWarning``: wrapping ``__init__``
    cannot retrofit a client that has already been constructed, so a module-level
    ``requests.Session()`` built during someone else's import is missed silently.
    """


# Only the optional extras are worth checking. httpx is always present by the
# time this module can be imported at all, because the package __init__ pulls in
# client.py, so its presence carries no signal — the SDK's own client sets the
# header itself instead.
_ORDERING_SENSITIVE = ("requests", "aiohttp")


def _warn_if_late() -> None:
    already = [name for name in _ORDERING_SENSITIVE if name in sys.modules]
    if not already:
        return

    warnings.warn(
        f"instrument() called after {', '.join(already)} "
        f"{'was' if len(already) == 1 else 'were'} already imported. Clients "
        "constructed before this point do not carry "
        f"{SOURCE_HEADER} and cannot be retrofitted. Call instrument() at the "
        "top of the service entrypoint, before importing anything that builds "
        "an HTTP client.",
        LateInstrumentationWarning,
        stacklevel=3,
    )


def instrument() -> None:
    """Wrap the installed HTTP clients. Idempotent; safe to call once at startup.

    Must run before any HTTP client is constructed — see
    :class:`LateInstrumentationWarning`.
    """
    global _instrumented
    if _instrumented:
        return

    _warn_if_late()

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
