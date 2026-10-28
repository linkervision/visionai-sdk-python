"""The relay wire format: the ADR's JSON examples parse on both sides, bad frames say why."""

import json

import pytest

from visionai_sdk_python.relay import (
    MAX_FRAME_BYTES,
    MAX_MESSAGE_CHARS,
    Error,
    FrameTooLarge,
    Heartbeat,
    InvalidFrame,
    Register,
    Registered,
    Request,
    Response,
    dump_frame,
    is_workspace_id,
    parse_agent_frame,
    parse_relay_frame,
)
from visionai_sdk_python.relay.examples import NAMES
from visionai_sdk_python.relay.examples import example_text as example


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("register", Register),
        ("heartbeat", Heartbeat),
        ("response", Response),
        ("response_text", Response),
        ("error_request", Error),
        ("error_connection", Error),
    ],
)
def test_the_relay_parses_the_adr_examples_an_agent_sends(name, kind):
    assert isinstance(parse_agent_frame(example(name)), kind)


@pytest.mark.parametrize(
    ("name", "kind"),
    [("registered", Registered), ("request", Request), ("error_connection", Error)],
)
def test_the_agent_parses_the_adr_examples_the_relay_sends(name, kind):
    assert isinstance(parse_relay_frame(example(name)), kind)


def test_register_example_carries_models_and_replicas():
    frame = parse_agent_frame(example("register"))

    assert frame.cluster_id == "aws-offline-edge"
    assert [w.id for w in frame.workspaces][2] == "QA Workspace"
    assert [(m.name, m.version, m.replicas) for m in frame.workspaces[0].models] == [
        ("cr2-8b", "4.5.0", 2),
        ("cr2-8b", "4.6.0", 1),
    ]


def test_request_example_fields():
    frame = parse_relay_frame(example("request"))

    assert (frame.method, frame.path, frame.timeout_s) == ("POST", "/api/chat", 100)
    assert frame.headers["X-Request-Source"] == "dataverse"
    assert frame.body["model"] == "cr2-8b"


@pytest.mark.parametrize(
    ("name", "kind"), [("register", Register), ("registered", Registered)]
)
def test_examples_round_trip(name, kind):
    data = json.loads(example(name))

    assert json.loads(dump_frame(kind.model_validate(data))) == data


def test_unknown_fields_are_ignored_on_both_sides():
    assert isinstance(parse_agent_frame('{"type": "heartbeat", "extra": 1}'), Heartbeat)
    assert isinstance(
        parse_relay_frame('{"type": "registered", "accepted": [], "x": 1}'), Registered
    )


def test_register_without_replicas_is_invalid():
    with pytest.raises(InvalidFrame):
        parse_agent_frame(
            '{"type": "register", "agent_id": "a", "cluster_id": "c",'
            ' "workspaces": [{"id": "x", "models": [{"name": "m", "version": "1"}]}]}'
        )


def test_unknown_reasons_and_codes_are_accepted_as_strings():
    registered = parse_relay_frame(
        '{"type": "registered", "accepted": [], "rejected": [{"id": "x", "reason": "new_reason"}]}'
    )
    error = parse_agent_frame('{"type": "error", "code": "new_code", "message": "m"}')

    assert registered.rejected[0].reason == "new_reason"
    assert error.code == "new_code"


@pytest.mark.parametrize(
    ("parse", "text", "message", "request_id"),
    [
        (
            parse_agent_frame,
            b'{"type": "heartbeat"}',
            "binary frames are not used",
            None,
        ),
        (parse_relay_frame, None, "binary frames are not used", None),
        (parse_agent_frame, "not json", "frame is not JSON", None),
        (parse_agent_frame, "[1, 2]", "frame is not a JSON object", None),
        (parse_agent_frame, '{"agent_id": "a"}', "frame has no type", None),
        (parse_agent_frame, '{"type": 3}', "frame has no type", None),
        (parse_agent_frame, '{"type": "ping"}', "unknown type 'ping'", None),
        (
            parse_agent_frame,
            '{"type": "register"}',
            "register fails validation: agent_id, cluster_id",
            None,
        ),
        (
            parse_agent_frame,
            '{"type": "response", "request_id": "r"}',
            "response fails validation: status",
            "r",
        ),
        (
            parse_relay_frame,
            '{"type": "ping", "request_id": "r1"}',
            "unknown type 'ping'",
            "r1",
        ),
        (
            parse_relay_frame,
            '{"type": "request", "request_id": "r2"}',
            "request fails validation: workspace_id, method, path, timeout_s",
            "r2",
        ),
        (
            parse_relay_frame,
            '{"type": "request", "request_id": 7}',
            "request fails validation: request_id, workspace_id, method, path, timeout_s",
            None,
        ),
    ],
)
def test_invalid_frames_name_the_problem_and_keep_the_request_id(
    parse, text, message, request_id
):
    with pytest.raises(InvalidFrame) as exc:
        parse(text)

    assert str(exc.value) == message
    assert exc.value.request_id == request_id


def test_dump_frame_drops_none_fields():
    assert dump_frame(Error(code="invalid_message", message="x")) == (
        '{"type":"error","code":"invalid_message","message":"x"}'
    )


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41", True),
        ("31A6638A-5C2E-4F1B-9A7D-0E3C6B8D2F41", False),
        ("31a6638a5c2e4f1b9a7d0e3c6b8d2f41", False),
        ("x31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41", False),
        ("QA Workspace", False),
        ("sc_test", False),
        ("dev3", False),
    ],
)
def test_workspace_id_format(value, ok):
    assert is_workspace_id(value) is ok


def test_every_shipped_example_is_valid_json_with_a_type():
    for name in NAMES:
        assert isinstance(json.loads(example(name))["type"], str), name


def test_an_error_message_is_cut_so_it_fits_a_frame():
    with pytest.raises(InvalidFrame) as exc:
        parse_agent_frame(json.dumps({"type": "x" * 1_000_000}))

    assert len(str(exc.value)) == MAX_MESSAGE_CHARS
    assert str(exc.value).endswith("…")


def test_an_oversized_request_id_is_dropped_from_the_error():
    frame = {"type": "request", "request_id": "r" * 1_000_000}
    with pytest.raises(InvalidFrame) as exc:
        parse_relay_frame(json.dumps(frame))

    assert exc.value.request_id is None
    assert str(exc.value).endswith("(request_id too long, dropped)")


def test_dump_frame_refuses_a_frame_the_receiver_would_close_on():
    frame = Response(request_id="r1", status=200, body="x" * MAX_FRAME_BYTES)

    with pytest.raises(FrameTooLarge) as exc:
        dump_frame(frame)

    assert exc.value.size > MAX_FRAME_BYTES
    assert dump_frame(Response(request_id="r1", status=200, body="x"))
