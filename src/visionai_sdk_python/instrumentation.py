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

**Forwarding an inherited origin (A-5).** By default this stamps the local
service's own identity — correct as long as a service is the true origin of
whatever VLM call it makes. If service A calls service B (both instrumented)
and B then calls VLM on A's behalf, B should forward A's identity rather than
stamp its own, the same problem ``visionai-vlm-scheduling-service`` solves for
its Redis queue hop by explicitly storing and re-attaching the header. Use
``current_origin()``/``origin_from_headers()`` from your own inbound-request
middleware::

    from visionai_sdk_python import instrumentation

    origin = instrumentation.origin_from_headers(incoming_request.headers)
    with instrumentation.current_origin(origin):
        ...  # handle the request; a client built in this scope inherits `origin`

This makes forwarding automatic for a client built fresh inside that scope
(``instrument()``'s injection already runs at construction time and now checks
the current scope first). A client built once at startup and reused across many
requests can't pick up a value that varies per request just by being
instrumented — merge ``source_headers()`` into the per-call headers on outbound
calls made in that scope instead::

    response = shared_client.get(url, headers={**instrumentation.source_headers(), **other_headers})

``source_headers()`` returns ``{"X-Request-Source": value}`` (inherited origin,
falling back to this service's own identity — the same resolution
``instrument()``'s automatic injection uses) or ``{}`` if there is nothing to
send. Per-call headers already override a client's defaults in requests/httpx/
aiohttp, so no extra mechanism is needed for that case. **Do not** pass
``get_current_origin()`` directly as a header value — it legitimately returns
``None`` whenever there is no inherited origin (the common case, meaning this
service is the origin), and both libraries handle a ``None`` header value
badly: ``requests`` silently drops it, discarding the client's own correctly-set
default in the process, and ``httpx`` raises ``TypeError``. See "A-5" in the
service-source-attribution plan.
"""

import gc
import sys
import warnings
from typing import Any

import wrapt

from ._source_header import (
    SOURCE_HEADER,
    _effective_source,
    current_origin,
    get_current_origin,
    origin_from_headers,
    source_headers,
)

__all__ = [
    "instrument",
    "uninstrument",
    "current_origin",
    "get_current_origin",
    "origin_from_headers",
    "source_headers",
    "LateInstrumentationWarning",
]

_instrumented = False


def _already_set(headers: Any) -> bool:
    pairs = list(headers.items()) if hasattr(headers, "items") else list(headers or [])
    return any(key.lower() == SOURCE_HEADER.lower() for key, _ in pairs)


def _set_after_init(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """For clients exposing a mutable public ``.headers`` (requests, httpx).

    Reads ``_effective_source()`` at construction time, so a client built inside
    a ``current_origin()`` scope picks up the inherited value rather than this
    service's own — see "Forwarding an inherited origin" above.
    """
    result = wrapped(*args, **kwargs)

    source = _effective_source()
    if source and not _already_set(instance.headers):
        instance.headers[SOURCE_HEADER] = source

    return result


def _set_via_init_kwarg(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """For aiohttp, whose default headers can only be set through ``__init__``.

    Passing a list of pairs rather than a dict keeps any duplicate headers the
    caller supplied. This is also the one injection vehicle ``aioresponses``
    can see, which a ``TraceConfig``-based approach cannot offer.
    """
    source = _effective_source()
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
    """Raised when instrument() runs after a client was already constructed.

    Wrapping ``__init__`` cannot retrofit a client built before this point, so
    it silently never carries ``X-Request-Source``, and this is the only signal
    that happened.

    This is *not* the same check ``gevent.monkey``'s ``MonkeyPatchWarning``
    does, despite the similar spirit. gevent's ``patch_ssl()`` *rebinds*
    ``ssl.SSLContext`` to a different class, so it scans already-imported
    modules for a stale reference to the pre-patch class (via ``gc.get_referrers``)
    — a real, forward-looking risk, since anything built later from that stale
    reference stays broken forever. We patch ``__init__`` in place; the class
    object's identity never changes, so a module holding
    ``from httpx import Client`` from before ``instrument()`` ran is completely
    fine for anything it constructs afterward — Python looks up ``__init__`` on
    the class at call time, and we mutated the class's own attribute. Our risk
    is purely backward-looking: an instance built *before* the patch landed. So
    instead we scan for already-*existing instances* via ``gc.get_objects()``,
    which is the check that actually matches our failure mode.
    """


def _owned_by_sdk_client(obj: Any) -> bool:
    """True if `obj` is the internal httpx client of our own Client/AsyncClient.

    Those get X-Request-Source from source_headers() directly in __init__,
    unconditionally — not through instrument()'s wrapt patching at all — so their
    attribution never depends on instrument()'s timing and this is not a gap.
    """
    from . import async_client as _async_client_module
    from . import client as _client_module

    for ref in gc.get_referrers(obj):
        if isinstance(ref, (_client_module.Client, _async_client_module.AsyncClient)):
            if getattr(ref, "_client", None) is obj:
                return True
    return False


def _find_preexisting_clients() -> dict[str, int]:
    """Count already-built instances of the classes instrument() is about to wrap.

    Runs once at instrument()-time, before patching; the one-time cost of a full
    heap scan is a startup-time concern, not a request-path one.
    """
    target_classes = []
    for module_name, class_name in (
        ("requests", "Session"),
        ("httpx", "Client"),
        ("httpx", "AsyncClient"),
        ("aiohttp", "ClientSession"),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            target_classes.append(getattr(module, class_name))

    if not target_classes:
        return {}

    counts: dict[str, int] = {}
    for obj in gc.get_objects():
        try:
            is_target = isinstance(obj, tuple(target_classes))
        except ReferenceError:
            continue
        if not is_target or _owned_by_sdk_client(obj):
            continue
        name = f"{type(obj).__module__}.{type(obj).__qualname__}"
        counts[name] = counts.get(name, 0) + 1

    return counts


def _warn_if_late() -> None:
    preexisting = _find_preexisting_clients()
    if not preexisting:
        return

    described = ", ".join(
        f"{name} (×{count})" if count > 1 else name
        for name, count in preexisting.items()
    )
    warnings.warn(
        f"instrument() called after these clients were already constructed: "
        f"{described}. They were built before instrument() could wrap their "
        f"__init__ and do not carry {SOURCE_HEADER}; clients built from now on "
        "are unaffected. Call instrument() at the top of the service "
        "entrypoint, before importing anything that builds an HTTP client.",
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
