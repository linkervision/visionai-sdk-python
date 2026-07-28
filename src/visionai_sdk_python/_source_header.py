"""Shared header-injection logic for the drop-in HTTP shims.

Every shim (``requests``, ``httpx``, ``aiohttp``) reads the same
``VISIONAI_SERVICE_SOURCE`` environment variable and merges the same
``X-Request-Source`` header the same way, so the behavior is defined once
here instead of three times.

Two entry points, for two different injection points:

- ``merge_source_headers`` -- for module-level functions with no
  persistent Session/Client (``requests.get``, ``httpx.post``, ...):
  there is only one set of headers to consider (whatever the caller
  passed to this specific call), so a pre-merge check against that one
  dict is enough.
- ``inject_source_header`` -- for ``Session.send()``/``Client.send()``
  overrides: by the time a request reaches ``send()``, the real library
  has already merged the Session/Client's own default headers with any
  per-call headers into one final, already-case-insensitive header view
  -- covering every path that produces a request object, including a
  caller who builds one manually and calls ``send()`` directly, bypassing
  ``request()``/``build_request()`` entirely. A single presence check
  against that final view is both simpler and more complete than trying
  to enumerate "per-call vs default" at an earlier point.
"""

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

SOURCE_ENV_VAR = "VISIONAI_SERVICE_SOURCE"
SOURCE_HEADER = "X-Request-Source"


def merge_source_headers(existing_headers: Any) -> Any:
    """Merge ``X-Request-Source`` into the caller's headers, if needed.

    The env var is read fresh on every call (a single dict lookup, not
    worth caching). If it is unset, or the caller already supplied
    ``X-Request-Source`` themselves (matched case-insensitively, since
    HTTP header names are case-insensitive but a plain dict key is not),
    ``existing_headers`` is returned completely untouched -- the exact
    same object, not a copy -- so callers who passed something other than
    a plain dict (a ``list[tuple[str, str]]`` with intentionally
    duplicated header names, a multidict, ...) don't get silently
    coerced into a dict and lose data on a call where this function was
    never going to change anything anyway.

    Only when an actual injection is needed does this build a new
    object: a dict copy for ``Mapping`` input, or the input's pairs plus
    one more pair for a non-``Mapping`` iterable of pairs. Note this
    still collapses duplicate keys in the rare case of injecting into a
    ``Mapping`` that itself holds duplicates (e.g. a hand-built
    multidict) -- fully preserving that would mean reconstructing the
    caller's exact container type, which isn't worth the complexity for
    how unusual duplicate *request* headers are in practice.

    Args:
        existing_headers: The headers the caller passed to this request,
            if any -- a ``Mapping``, an iterable of ``(key, value)``
            pairs, or ``None``.

    Returns:
        ``existing_headers`` unchanged if nothing needs to change; a dict
        or list of pairs with the header added otherwise; ``{}`` if
        ``existing_headers`` was ``None`` and nothing needs to change.
    """
    if existing_headers is None:
        source = os.environ.get(SOURCE_ENV_VAR)
        return {SOURCE_HEADER: source} if source else {}

    pairs: list[tuple[str, str]] = (
        list(existing_headers.items())
        if isinstance(existing_headers, Mapping)
        else list(existing_headers)
    )

    source = os.environ.get(SOURCE_ENV_VAR)
    if not source or any(key.lower() == SOURCE_HEADER.lower() for key, _ in pairs):
        return existing_headers

    if isinstance(existing_headers, Mapping):
        merged = dict(existing_headers)
        merged[SOURCE_HEADER] = source
        return merged

    return [*pairs, (SOURCE_HEADER, source)]


def inject_source_header(headers: MutableMapping[str, str]) -> None:
    """Set ``X-Request-Source`` on ``headers`` in place, if not already present.

    Meant for a request's final, already-merged header mapping (e.g.
    ``PreparedRequest.headers``, ``httpx.Request.headers``) at the point
    just before it's actually sent -- see module docstring for why this
    is a single check rather than the two-source merge
    ``merge_source_headers`` does.
    """
    if any(key.lower() == SOURCE_HEADER.lower() for key in headers):
        return

    source = os.environ.get(SOURCE_ENV_VAR)
    if source:
        headers[SOURCE_HEADER] = source
