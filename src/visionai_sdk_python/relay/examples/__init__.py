"""The ADR's JSON examples, shipped so every implementation tests against the same frames.

``example("register")`` is the parsed object, ``example_text("register")`` the wire text.
"""

import json
from importlib.resources import files

NAMES = (
    "register",
    "registered",
    "heartbeat",
    "request",
    "response",
    "response_text",
    "error_request",
    "error_connection",
)


def example_text(name: str) -> str:
    """One ADR example as the text that goes over the wire."""
    return (files(__package__) / f"{name}.json").read_text(encoding="utf-8")


def example(name: str) -> dict:
    """One ADR example as a dict, for building or comparing frames in tests."""
    return json.loads(example_text(name))
