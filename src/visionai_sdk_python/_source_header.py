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

SOURCE_ENV_VAR = "VISIONAI_SERVICE_SOURCE"
SOURCE_HEADER = "X-Request-Source"


def merge_source_headers(existing_headers: Mapping[str, str] | None) -> dict[str, str]:
    """Merge ``X-Request-Source`` into a copy of the caller's headers.

    The env var is read fresh on every call (a single dict lookup, not
    worth caching). If it is unset, or the caller already supplied
    ``X-Request-Source`` themselves (matched case-insensitively, since
    HTTP header names are case-insensitive but a plain dict key is not),
    the caller's headers are returned unchanged.

    Args:
        existing_headers: The headers the caller passed to this request,
            if any.

    Returns:
        A new dict of headers to send, never ``None``.
    """
    merged = dict(existing_headers) if existing_headers else {}

    source = os.environ.get(SOURCE_ENV_VAR)
    if not source:
        return merged

    if any(key.lower() == SOURCE_HEADER.lower() for key in merged):
        return merged

    merged[SOURCE_HEADER] = source
    return merged


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
