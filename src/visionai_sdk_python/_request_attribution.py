"""Per-request X-Request-Source merging for the SDK's own Client/AsyncClient.

Merges fresh on every call (not once at construction) so a later
current_origin() scope (A-5) is picked up, and tags the injection via
INJECTED_EXTENSION_KEY so instrumentation.py's per-request httpx scoping
recognizes it as ours -- rather than a caller-supplied value it must never
touch -- if instrument() is also active on the same process.
"""

from typing import Any

from ._source_header import INJECTED_EXTENSION_KEY, SOURCE_HEADER, source_headers


def merge_request_attribution(kwargs: dict[str, Any]) -> None:
    """Mutate ``kwargs`` in place: add ``headers``/``extensions`` if not already set."""
    headers = kwargs.get("headers") or {}
    already_set = any(key.lower() == SOURCE_HEADER.lower() for key in headers)
    if already_set:
        return

    source = source_headers()
    if not source:
        return

    kwargs["headers"] = {**headers, **source}
    kwargs["extensions"] = {
        **(kwargs.get("extensions") or {}),
        INJECTED_EXTENSION_KEY: True,
    }
