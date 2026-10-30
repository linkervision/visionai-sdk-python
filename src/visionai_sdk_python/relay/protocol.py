"""Wire format between platform-relay and platform-edge-agent.

The spec is ``docs/relay-protocol.md`` in this repository (the decisions behind it:
platform-service's ADR 0001); this module is the one implementation both sides import. Frames are JSON objects told apart by
``type``. Receivers ignore unknown fields, and the string-typed ``reason`` /
``code`` fields accept values they do not know, so either side can be upgraded
first.
"""

from __future__ import annotations

import json
import re
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

SUBPROTOCOL = "platform-relay.v1"
# The first frame after the upgrade must be `register`, this soon; else the relay closes 1008.
REGISTER_DEADLINE_S = 10.0
HEARTBEAT_INTERVAL_S = 30.0
# No frame for this long: the relay closes 4001 and forgets the connection.
STALE_AFTER_S = 90.0
# Both sides refuse to send a larger frame; the relay refuses a larger chat body (413).
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_BODY_BYTES = 15 * 1024 * 1024
# Timeouts nest: the agent waits for VS, the relay waits a little longer for the agent.
AGENT_TIMEOUT_S = 100
RELAY_TIMEOUT_S = 110.0
# An `error` echoes parts of the peer's frame; cut them so the error fits a frame.
MAX_MESSAGE_CHARS = 200

# Platform workspace ids: lowercase hyphenated UUIDs.
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def is_workspace_id(value: str) -> bool:
    return UUID_RE.fullmatch(value) is not None


class CloseCode(IntEnum):
    NORMAL = 1000
    GOING_AWAY = 1001
    POLICY_VIOLATION = 1008
    TOO_BIG = 1009
    REPLACED = 4000
    STALE = 4001


class RejectReason(StrEnum):
    INVALID_ID = "invalid_id"
    CLUSTER_MISMATCH = "cluster_mismatch"
    PLATFORM_UNAVAILABLE = "platform_unavailable"


class ErrorCode(StrEnum):
    INVALID_MESSAGE = "invalid_message"
    UNKNOWN_WORKSPACE = "unknown_workspace"
    PATH_NOT_ALLOWED = "path_not_allowed"
    UPSTREAM_UNREACHABLE = "upstream_unreachable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_TOO_LARGE = "upstream_too_large"


class Frame(BaseModel):
    # Receivers ignore unknown fields.
    model_config = ConfigDict(extra="ignore")


class ModelEntry(Frame):
    name: str
    version: str
    replicas: int = Field(ge=0)


class WorkspaceEntry(Frame):
    id: str
    models: list[ModelEntry] = Field(default_factory=list)


class Register(Frame):
    type: Literal["register"] = "register"
    agent_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    workspaces: list[WorkspaceEntry] = Field(default_factory=list)


class Rejected(Frame):
    id: str
    reason: str  # a RejectReason; a plain str so an older agent accepts new reasons


class Registered(Frame):
    type: Literal["registered"] = "registered"
    accepted: list[str] = Field(default_factory=list)
    rejected: list[Rejected] = Field(default_factory=list)


class Heartbeat(Frame):
    type: Literal["heartbeat"] = "heartbeat"


class Request(Frame):
    type: Literal["request"] = "request"
    request_id: str
    workspace_id: str
    method: str
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict | list | str | None = None
    timeout_s: int = Field(gt=0)


class Response(Frame):
    type: Literal["response"] = "response"
    request_id: str
    status: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict | list | str | None = None


class Error(Frame):
    type: Literal["error"] = "error"
    request_id: str | None = None
    code: str  # an ErrorCode; a plain str so an older receiver accepts new codes
    message: str


# What each side receives.
AgentFrame = Register | Heartbeat | Response | Error  # agent -> relay
RelayFrame = Registered | Request | Error  # relay -> agent
_agent_frame = TypeAdapter(Annotated[AgentFrame, Field(discriminator="type")])
_relay_frame = TypeAdapter(Annotated[RelayFrame, Field(discriminator="type")])


class InvalidFrame(ValueError):
    """A frame the receiver cannot read; the text is safe to send back in an ``error``.

    Parts of the text come from the peer's frame, so it is cut at ``MAX_MESSAGE_CHARS``
    and a ``request_id`` longer than that is dropped.
    """

    def __init__(self, message: str, request_id: str | None = None) -> None:
        if len(message) > MAX_MESSAGE_CHARS:
            message = message[: MAX_MESSAGE_CHARS - 1] + "…"
        if request_id is not None and len(request_id) > MAX_MESSAGE_CHARS:
            request_id = None
            message += " (request_id too long, dropped)"
        super().__init__(message)
        self.request_id = request_id


class FrameTooLarge(ValueError):
    """The encoded frame exceeds ``MAX_FRAME_BYTES``; the receiver would close with 1009."""

    def __init__(self, size: int) -> None:
        super().__init__(f"frame is {size} bytes; the limit is {MAX_FRAME_BYTES}")
        self.size = size


def parse_agent_frame(message: str | bytes | None) -> AgentFrame:
    """A frame the relay received from an agent.

    ``message`` is what the WebSocket library handed over: ``str`` for a text frame,
    ``bytes`` (or Starlette's ``None``) for a binary one, which the protocol does
    not use and is an ``InvalidFrame`` like any other unreadable frame.
    """
    return _parse(message, _agent_frame)


def parse_relay_frame(message: str | bytes | None) -> RelayFrame:
    """A frame the agent received from the relay; see ``parse_agent_frame``."""
    return _parse(message, _relay_frame)


def _parse(message: str | bytes | None, adapter: TypeAdapter):
    if not isinstance(message, str):
        # json.loads() would happily read JSON out of a binary frame; say no once here.
        raise InvalidFrame("binary frames are not used")
    try:
        data = json.loads(message)
    except ValueError:
        raise InvalidFrame("frame is not JSON") from None
    if not isinstance(data, dict):
        raise InvalidFrame("frame is not a JSON object")
    kind = data.get("type")
    if not isinstance(kind, str):
        raise InvalidFrame("frame has no type")
    request_id = data.get("request_id")
    request_id = request_id if isinstance(request_id, str) else None
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        if any(err["type"] == "union_tag_invalid" for err in exc.errors()):
            raise InvalidFrame(f"unknown type {kind!r}", request_id) from None
        fields = ", ".join(
            ".".join(str(part) for part in err["loc"][1:]) or "body"
            for err in exc.errors()
        )
        raise InvalidFrame(f"{kind} fails validation: {fields}", request_id) from None


def dump_frame(frame: BaseModel) -> str:
    """The frame as JSON text, or ``FrameTooLarge`` when a receiver would refuse it."""
    text = frame.model_dump_json(exclude_none=True)
    size = len(text.encode())
    if size > MAX_FRAME_BYTES:
        raise FrameTooLarge(size)
    return text
