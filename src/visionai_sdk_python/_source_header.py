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
