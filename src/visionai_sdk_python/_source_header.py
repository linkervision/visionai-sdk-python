"""Reads and validates the VISIONAI_SERVICE_SOURCE env var for X-Request-Source.

Also holds the request-scoped "inherited origin" mechanism (A-5): a service that
is itself called by another SDK-instrumented service, and forwards that caller's
identity instead of stamping its own, uses ``current_origin()``/
``origin_from_headers()`` — see ``instrumentation.py`` for the public re-export
and the README for usage.
"""

import contextlib
import contextvars
import os
import re
import warnings
from collections.abc import Generator
from typing import Any

SOURCE_ENV_VAR = "VISIONAI_SERVICE_SOURCE"
SOURCE_HEADER = "X-Request-Source"

# Shared "did *we* set this header" marker, read by instrumentation.py's
# per-request httpx scoping and set here so client.py/async_client.py can
# participate without importing instrumentation.py (which pulls in wrapt).
INJECTED_EXTENSION_KEY = "visionai_sdk_python.source_header_injected"

# Printable ASCII with no leading/trailing space. Stricter than any single
# library's own check, so a value that passes here is safe to hand to requests,
# httpx and aiohttp alike.
_VALID_SOURCE = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?")


def _validate(raw: str) -> str | None:
    """Strip and validate a candidate header value, warning and dropping if unusable.

    Attribution is best-effort telemetry, so a malformed value — ours or an
    inherited one from an upstream caller — is dropped with a warning rather
    than allowed to break the caller's request.
    """
    value = raw.strip()
    if not value:
        return None

    if not _VALID_SOURCE.fullmatch(value):
        warnings.warn(
            f"{raw!r} is not a valid HTTP header value; skipping {SOURCE_HEADER} "
            "injection.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    return value


def _source_value() -> str | None:
    """This service's own identity from VISIONAI_SERVICE_SOURCE, or None if unset."""
    raw = os.environ.get(SOURCE_ENV_VAR)
    if raw is None:
        return None
    return _validate(raw)


_current_origin: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "visionai_sdk_current_origin", default=None
)


def _decode(value: str | bytes) -> str:
    # HTTP header bytes are ISO-8859-1 per RFC 7230 (and the ASGI spec spells
    # this out explicitly for scope["headers"]) -- latin-1 decodes any byte
    # sequence without raising, unlike utf-8.
    return value.decode("latin-1") if isinstance(value, bytes) else value


def origin_from_headers(headers: Any) -> str | None:
    """Read an inbound X-Request-Source value, case-insensitively.

    ``headers`` may be a mapping (dict, CIMultiDict, a framework's own headers
    object exposing ``.items()``) or an iterable of ``(key, value)`` pairs --
    including raw ``bytes`` keys/values, as in ASGI's ``scope["headers"]``.
    Returns None if absent — meaning the caller of the current request did not
    already attribute it, so this service is the origin of whatever follows.
    """
    pairs = list(headers.items()) if hasattr(headers, "items") else list(headers or [])
    for key, value in pairs:
        if _decode(key).lower() == SOURCE_HEADER.lower():
            return _decode(value)
    return None


@contextlib.contextmanager
def current_origin(value: str | None) -> Generator[None, None, None]:
    """Scope the origin forwarded on outbound calls made while handling one request.

    Call this from your own inbound-request middleware, wrapping the handling of
    each request::

        origin = origin_from_headers(incoming_request.headers)
        with current_origin(origin):
            ...  # handle the request, including any outbound VLM calls

    If ``value`` is None (no inbound X-Request-Source), outbound calls made in
    this scope fall back to this service's own VISIONAI_SERVICE_SOURCE — i.e.
    this service becomes the origin for whatever follows, exactly as
    ``instrument()`` already behaves without this API. Safe under concurrent
    requests: backed by ``contextvars``, isolated per asyncio task and per
    thread, and the previous value is restored on exit so nesting works.
    """
    token = _current_origin.set(value)
    try:
        yield
    finally:
        _current_origin.reset(token)


def get_current_origin() -> str | None:
    """The value set by the nearest enclosing ``current_origin()`` scope.

    None outside any such scope, or if the scoped value fails the same
    validation VISIONAI_SERVICE_SOURCE is held to.
    """
    value = _current_origin.get()
    if value is None:
        return None
    return _validate(value)


def _effective_source() -> str | None:
    """What to inject: last hop (our own identity) unless A-5 set an inherited origin."""
    inherited = get_current_origin()
    if inherited is not None:
        return inherited
    return _source_value()


def source_headers() -> dict[str, str]:
    """Default headers for the SDK's own clients — empty when nothing to inject."""
    source = _effective_source()
    return {SOURCE_HEADER: source} if source else {}
