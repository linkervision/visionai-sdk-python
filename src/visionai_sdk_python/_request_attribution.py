"""Per-request X-Request-Source merging for the SDK's own Client/AsyncClient.

Merges fresh on every call (not once at construction) so a later
current_origin() scope (A-5) is picked up, and tags the injection via
INJECTED_EXTENSION_KEY so instrumentation.py's per-request httpx scoping
recognizes it as ours -- rather than a caller-supplied value it must never
touch -- if instrument() is also active on the same process.
"""

from typing import Any

from ._source_header import (
    INJECTED_EXTENSION_KEY,
    SOURCE_HEADER,
    _decode,
    source_headers,
)


def merge_request_attribution(kwargs: dict[str, Any]) -> None:
    """Mutate ``kwargs`` in place: add ``headers``/``extensions`` if not already set.

    Materializes ``headers`` into a pairs list rather than a dict -- httpx
    accepts duplicate headers as legitimate (a dict would silently collapse
    them to one), and appending a pair rather than rebuilding a dict keeps
    them intact. Always writes the materialized pairs back to
    ``kwargs["headers"]``, even on an early return: if the caller passed a
    one-shot iterable (e.g. a generator), inspecting it above already
    consumed it, so leaving the original object in place would hand the
    caller's HTTP library something already exhausted, silently dropping
    every header -- including the caller's own X-Request-Source.
    """
    headers = kwargs.get("headers")
    pairs = list(headers.items()) if hasattr(headers, "items") else list(headers or [])
    already_set = any(_decode(key).lower() == SOURCE_HEADER.lower() for key, _ in pairs)

    if not already_set:
        source = source_headers().get(SOURCE_HEADER)
        if source:
            pairs.append((SOURCE_HEADER, source))
            kwargs["extensions"] = {
                **(kwargs.get("extensions") or {}),
                INJECTED_EXTENSION_KEY: True,
            }

    kwargs["headers"] = pairs
