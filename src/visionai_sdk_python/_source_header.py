"""Shared X-Request-Source header logic for the requests/httpx/aiohttp shims.

``merge_source_headers`` is for module-level calls (no persistent
Session/Client); ``inject_source_header`` is for Session/Client ``send()``
overrides, where headers are already fully merged.
"""

import os
import re
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any

SOURCE_ENV_VAR = "VISIONAI_SERVICE_SOURCE"
SOURCE_HEADER = "X-Request-Source"

# Printable ASCII with no leading/trailing space. Stricter than any single
# library's own check, so a value that passes here is safe to hand to requests,
# httpx and aiohttp alike.
_VALID_SOURCE = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?")


def _source_value() -> str | None:
    """Read the env var, or None if unset or unusable as a header value.

    Attribution is best-effort telemetry, so a malformed value is dropped with a
    warning rather than allowed to break the caller's request.
    """
    raw = os.environ.get(SOURCE_ENV_VAR)
    if raw is None:
        return None

    value = raw.strip()
    if not value:
        return None

    if not _VALID_SOURCE.fullmatch(value):
        warnings.warn(
            f"{SOURCE_ENV_VAR}={raw!r} is not a valid HTTP header value; "
            f"skipping {SOURCE_HEADER} injection.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    return value


def source_headers() -> dict[str, str]:
    """Default headers for the SDK's own clients — empty when the env var is unset."""
    source = _source_value()
    return {SOURCE_HEADER: source} if source else {}


def merge_source_headers(existing_headers: Any) -> Any:
    """Merge X-Request-Source into caller headers; no-op if unset or already present.

    Returns the same object untouched when no change is needed, so
    non-dict inputs (list of pairs, multidict) aren't silently coerced.
    """
    if existing_headers is None:
        source = _source_value()
        return {SOURCE_HEADER: source} if source else {}

    pairs: list[tuple[str, str]] = (
        list(existing_headers.items())
        if isinstance(existing_headers, Mapping)
        else list(existing_headers)
    )

    source = _source_value()
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

    source = _source_value()
    if source:
        headers[SOURCE_HEADER] = source
