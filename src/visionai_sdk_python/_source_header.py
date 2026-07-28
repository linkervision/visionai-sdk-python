"""Shared header-injection logic for the drop-in HTTP shims.

Every shim (``requests``, ``httpx``, ``aiohttp``) reads the same
``VISIONAI_SERVICE_SOURCE`` environment variable and merges the same
``X-Request-Source`` header the same way, so the behavior is defined once
here instead of three times.
"""

import os
from collections.abc import Mapping

SOURCE_ENV_VAR = "VISIONAI_SERVICE_SOURCE"
SOURCE_HEADER = "X-Request-Source"


def merge_source_headers(
    existing_headers: Mapping[str, str] | None,
    default_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge ``X-Request-Source`` into a copy of the caller's headers.

    The env var is read fresh on every call (a single dict lookup, not
    worth caching). If it is unset, or the caller already supplied
    ``X-Request-Source`` themselves -- either on this specific call
    (``existing_headers``) or as a default configured on the Session/Client
    itself (``default_headers``, e.g. ``requests.Session.headers`` or
    ``httpx.Client(headers=...)``) -- matched case-insensitively, since HTTP
    header names are case-insensitive but a plain dict key is not, the
    caller's headers are returned unchanged.

    ``default_headers`` matters because callers of ``Session``/``Client``
    typically configure a header once at construction time rather than on
    every individual call; without checking it too, this function can't
    tell "caller didn't set anything" apart from "caller set it at the
    Session/Client level," and would inject the env value as a per-call
    header that then wins over the caller's own default in both requests'
    and httpx's own header-merge precedence.

    Args:
        existing_headers: The headers the caller passed to this specific
            request, if any.
        default_headers: Headers already configured as defaults on the
            underlying Session/Client, if any. Only consulted to decide
            whether to inject -- never copied into the returned dict, since
            the real library already merges its own defaults separately.

    Returns:
        A new dict of headers to send, never ``None``.
    """
    merged = dict(existing_headers) if existing_headers else {}

    source = os.environ.get(SOURCE_ENV_VAR)
    if not source:
        return merged

    if any(key.lower() == SOURCE_HEADER.lower() for key in merged):
        return merged

    if default_headers and any(
        key.lower() == SOURCE_HEADER.lower() for key in default_headers
    ):
        return merged

    merged[SOURCE_HEADER] = source
    return merged
