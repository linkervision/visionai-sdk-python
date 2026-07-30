"""Reads and validates the VISIONAI_SERVICE_SOURCE env var for X-Request-Source."""

import os
import re
import warnings

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
