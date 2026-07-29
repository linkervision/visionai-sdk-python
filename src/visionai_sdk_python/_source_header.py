"""Shared X-Request-Source header logic for the requests/httpx/aiohttp shims.

``merge_source_headers`` is for module-level calls (no persistent
Session/Client); ``inject_source_header`` is for Session/Client ``send()``
overrides, where headers are already fully merged.
"""

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

SOURCE_ENV_VAR = "VISIONAI_SERVICE_SOURCE"
SOURCE_HEADER = "X-Request-Source"


def merge_source_headers(existing_headers: Any) -> Any:
    """Merge X-Request-Source into caller headers; no-op if unset or already present.

    Returns the same object untouched when no change is needed, so
    non-dict inputs (list of pairs, multidict) aren't silently coerced.
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
    """Set X-Request-Source on an already-merged headers mapping, in place, if absent."""
    if any(key.lower() == SOURCE_HEADER.lower() for key in headers):
        return

    source = os.environ.get(SOURCE_ENV_VAR)
    if source:
        headers[SOURCE_HEADER] = source
