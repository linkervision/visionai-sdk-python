# platform-relay ↔ platform-edge-agent wire format, `platform-relay.v1`

What goes over the WebSocket between platform-relay (platform-service) and platform-edge-agent
(visionai-vlm-scheduling-service). `visionai_sdk_python.relay` implements it, the models, the
constants and the parsers, and both sides import that module instead of keeping a copy. The JSON
examples below ship as `visionai_sdk_python.relay.examples` and are the fixtures of the SDK's, the
relay's and the agent's tests. A protocol change edits this file and the module in one PR; the two
consumers then bump the SDK.

Why the protocol looks like this, how the relay checks a registration against Platform, how it
routes and what its HTTP API answers callers are decisions, recorded in platform-service's
`docs/adr/0001-relay-protocol.md`.

## 1. Connection

The agent connects to `wss://<platform_domain>/relay/ws`. The HTTP upgrade carries the agent's
identity and the protocol version:

```http
GET /relay/ws HTTP/1.1
Host: <platform_domain>
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Version: 13
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Protocol: platform-relay.v1
Authorization: Bearer eyJhbGciOi…

HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: platform-relay.v1
```

- `Authorization` is a platform M2M token (client credentials). A refused upgrade is an HTTP
  status, not a close code: 400 when the relay knows none of the offered subprotocols, 401 when the
  token is not accepted (the agent fetches a new token, once, and reconnects), 503 when the relay
  could not validate it (the agent retries with the same token).
- `Sec-WebSocket-Protocol` carries the protocol version (§6).
- After `101` the connection is the identity. Messages carry no agent or tenant fields.

Transport:

- Text frames only, one JSON object per frame, UTF-8. A binary frame is answered with
  `error invalid_message`.
- Maximum frame size 16 MiB in both directions; both sides check the encoded frame before sending
  (`dump_frame()` refuses a larger one).
- WebSocket ping every 20 s, pong timeout 60 s, on both sides. Large frames on slow links delay
  pongs, so the timeout is not the library default.

## 2. Messages

Six message types, told apart by `type`. Optional fields may be omitted or `null`. Receivers ignore
unknown fields.

| `type` | Sender | Fields |
|---|---|---|
| `register` | agent | `agent_id`, `cluster_id`, `workspaces` |
| `registered` | relay | `accepted`, `rejected` |
| `heartbeat` | agent | none |
| `request` | relay | `request_id`, `workspace_id`, `method`, `path`, `headers`, `body`, `timeout_s` |
| `response` | agent | `request_id`, `status`, `headers`, `body` |
| `error` | either | `request_id` (optional), `code`, `message` |

| Field | Type | Meaning |
|---|---|---|
| `agent_id` | string | The agent's pod name. With `cluster_id` it identifies a reconnecting agent (§3). |
| `cluster_id` | string | The agent's site, from its own configuration: the value vlm-mgmt reports as `VLM_CLUSTER_NAME` and Platform stores in `Node.cluster`. Required. |
| `workspaces` | array of object | `{"id", "models"}` per workspace the agent serves. `id` is the workspace's Platform id, a lowercase hyphenated UUID. Complete list; replaces the previous one. |
| `models` | array of object | `{"name", "version", "replicas"}` per model the workspace's VS lists as available (§3). Empty when it lists none. |
| `replicas` | integer | Required. Available nodes listing that model and version (§3); weights the relay's routing. |
| `accepted` | array of string | Workspace ids from `workspaces` the relay registered. |
| `rejected` | array of object | `{"id", "reason"}` per refused workspace: the workspace id and a reason from §5. |
| `request_id` | string | UUID v4 minted by the relay. `response` and `error` echo it. |
| `workspace_id` | string | Target workspace of a `request`. |
| `method`, `path` | string | The caller's HTTP method and path below `/<workspace_id>`, query string included. |
| `headers` | object | `request`: `Authorization`, `X-Request-Source`, `Content-Type` only. `response`: `Content-Type` only. Names are case-insensitive. |
| `body` | object, string or null | The HTTP body: JSON as an object, anything else as a string, `null` when there is none. |
| `timeout_s` | integer | Seconds the agent waits for VS. |
| `status` | integer | VS's HTTP status code, unchanged. |
| `code`, `message` | string | Error code (§5) and a human-readable detail, at most 200 characters. |

Examples. Image data and tokens are truncated; the `request` body is what Dataverse sends today.

`register`, agent → relay, first frame after `101`:

```json
{"type": "register",
 "agent_id": "platform-edge-agent-7d9f4b6c8-xk2pq",
 "cluster_id": "aws-offline-edge",
 "workspaces": [{"id": "31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41",
                 "models": [{"name": "cr2-8b", "version": "4.5.0", "replicas": 2},
                            {"name": "cr2-8b", "version": "4.6.0", "replicas": 1}]},
                {"id": "7589d018-2b4c-4e6f-8a1d-3c5e7f9b1d23", "models": []},
                {"id": "QA Workspace", "models": []}]}
```

`registered`, relay → agent. Platform lists no node of `aws-offline-edge` for `7589…`; `QA Workspace`
is not a UUID:

```json
{"type": "registered",
 "accepted": ["31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41"],
 "rejected": [{"id": "7589d018-2b4c-4e6f-8a1d-3c5e7f9b1d23", "reason": "cluster_mismatch"},
              {"id": "QA Workspace", "reason": "invalid_id"}]}
```

`heartbeat`, agent → relay, every 30 s:

```json
{"type": "heartbeat"}
```

`request`, relay → agent:

```json
{"type": "request",
 "request_id": "7f3c2a1e-9d4b-4c8a-b2e6-1f5d3a7c9e0b",
 "workspace_id": "31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41",
 "method": "POST",
 "path": "/api/chat",
 "headers": {"Authorization": "Bearer eyJhbGciOi…",
             "X-Request-Source": "dataverse",
             "Content-Type": "application/json"},
 "body": {"img": "data:image/jpeg;base64,/9j/4AAQ…",
          "prompt": "Describe the scene.",
          "temperature": 0.2, "max_tokens": 500, "top_p": 0.7,
          "stream": false, "use_cache": true, "num_beams": 1,
          "model": "cr2-8b", "model_version": null,
          "api_endpoint": null, "hook": null,
          "keep_alive": true, "use_response_postprocess": true,
          "workspace_id": "31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41"},
 "timeout_s": 100}
```

`response`, agent → relay. The same shape carries VS's 408, 422 and 500 answers:

```json
{"type": "response",
 "request_id": "7f3c2a1e-9d4b-4c8a-b2e6-1f5d3a7c9e0b",
 "status": 200,
 "headers": {"Content-Type": "application/json"},
 "body": {"chat_id": "high:1760424000123-3f9a1c7e",
          "status": "completed",
          "message": "{\"description\": \"A forklift in a warehouse aisle.\"}",
          "dispatch": {"model_name": "cr2-8b", "model_version": "4.5.0",
                       "api_endpoint": "http://cr2-8b:8000"}}}
```

`response` with a non-JSON body, for example an unhandled 500 from VS:

```json
{"type": "response",
 "request_id": "7f3c2a1e-9d4b-4c8a-b2e6-1f5d3a7c9e0b",
 "status": 500,
 "headers": {"Content-Type": "text/plain; charset=utf-8"},
 "body": "Internal Server Error"}
```

`error` about one request, agent → relay:

```json
{"type": "error",
 "request_id": "7f3c2a1e-9d4b-4c8a-b2e6-1f5d3a7c9e0b",
 "code": "upstream_unreachable",
 "message": "connect to vlm-scheduling-service-backend.31a6638a-5c2e-4f1b-9a7d-0e3c6b8d2f41.svc.cluster.local:80 failed"}
```

`error` about the connection, relay → agent, no `request_id`:

```json
{"type": "error",
 "code": "invalid_message",
 "message": "unknown type 'ping'"}
```

## 3. Registration and liveness

- The first frame after `101` MUST be `register`, within 10 s. Otherwise the relay closes with 1008.
- `workspaces` is every workspace the agent serves; an empty list is valid. `models` is what that
  workspace's VS reports on `GET /mgmt/nodes`: the `model_name` / `model_version` pairs of nodes
  with `status: available`, and in `replicas` how many such nodes list each pair.
- The relay answers every `register` with `registered`, checking each workspace on its own (reasons
  in §5). The connection stays open whatever the outcome.
- The agent re-sends `register` when its list changes, and on its next poll after a
  `platform_unavailable` rejection. The new list replaces the previous one; an agent shutting down
  may send an empty list first so no new `request` reaches it while it drains.
- A new connection with the same `cluster_id` and `agent_id` replaces the old one: the relay closes
  the old connection with 4000.
- The agent sends `heartbeat` every 30 s; the relay does not reply. No frame for 90 s: the relay
  closes with 4001.
- Shutdown: on SIGTERM the agent finishes in-flight requests, 10 s at most, then closes with 1001.
  The relay fails the requests still open on a connection that closes, whatever the code.

## 4. Request and response

The relay wraps the caller's HTTP request; the agent replays it against VS. The shape is generic
HTTP so further paths can be added without a protocol change.

- Several requests may be in flight on one connection. Responses may arrive in any order; the
  `request_id` pairs them.
- The agent forwards the three allowed headers unchanged and adds `X-Request-ID: <request_id>`.
  Nothing else is added.
- `timeout_s` is the agent's HTTP timeout for the call. Chat uses 100.
- Paths allowed in v1: `POST /api/chat`. Anything else: `error path_not_allowed`.
- The agent answers 422 without calling VS when the body has `stream: true`: a response is one
  frame, so SSE cannot be carried.
- The agent always answers: VS unreachable → `error upstream_unreachable`; no answer within
  `timeout_s` → `error upstream_timeout`; a `response` that would not fit a frame →
  `error upstream_too_large`; any other body → `response` with the body as a string.
- Neither side logs `Authorization` or `body.img` (linker-convention `conventions/logging.md` §6).

One chat, end to end:

```text
agent → relay   HTTP upgrade, Authorization + Sec-WebSocket-Protocol: platform-relay.v1
relay → agent   101 Switching Protocols
agent → relay   {"type": "register", "agent_id": "…", "cluster_id": "aws-offline-edge", "workspaces": [{"id": "31a6…", "models": [{"name": "cr2-8b", "version": "4.5.0", "replicas": 2}]}, {"id": "7589…", "models": []}]}
relay → agent   {"type": "registered", "accepted": ["31a6…", "7589…"], "rejected": []}
                … Dataverse: POST http://platform-relay/31a6…/api/chat
relay → agent   {"type": "request", "request_id": "7f3c…", "workspace_id": "31a6…", "method": "POST", "path": "/api/chat", …}
                … agent: POST http://vlm-scheduling-service-backend.31a6….svc.cluster.local/api/chat
agent → relay   {"type": "response", "request_id": "7f3c…", "status": 200, …}
                … relay: 200 with VS's body back to Dataverse
agent → relay   {"type": "heartbeat"}                      every 30 s throughout
```

## 5. Errors and close codes

An `error` that carries a `request_id` ends that request at once. An `error` without
`request_id` is logged and the connection stays open.

| `code` | Sender | When |
|---|---|---|
| `invalid_message` | either | Frame is binary, not JSON, has no `type`, an unknown `type`, or fails validation |
| `unknown_workspace` | agent | `workspace_id` is not in the agent's current list |
| `path_not_allowed` | agent | `method` + `path` outside §4 |
| `upstream_unreachable` | agent | Connection to VS failed, or VS's answer could not be read |
| `upstream_timeout` | agent | VS did not answer within `timeout_s` |
| `upstream_too_large` | agent | VS answered, but the `response` frame would be over 16 MiB |
| any other value | either | Added by a later version |

`reason` values in `registered.rejected`:

| `reason` | When |
|---|---|
| `invalid_id` | `id` is not a lowercase hyphenated UUID, or appears twice |
| `cluster_mismatch` | Platform lists no node of `cluster_id` for that workspace |
| `platform_unavailable` | The relay could not ask Platform; the agent's next `register` retries |

Close codes:

| Code | Sender | When |
|---|---|---|
| 1000 | either | Normal close |
| 1001 | either | Shutting down. Agents reconnect after a relay 1001; the relay drops the entry after an agent 1001 |
| 1008 | relay | Protocol violation: no `register` within 10 s, or a first frame that is not `register` |
| 1009 | either | Frame over 16 MiB. Both sides enforce the limit before sending, so this is a bug |
| 4000 | relay | Replaced by a new connection with the same `cluster_id` and `agent_id` |
| 4001 | relay | No frame for 90 s |

HTTP 400, 401 and 503 at the upgrade (§1) are not close codes; the connection never opens.

## 6. Versioning

- The version is the WebSocket subprotocol name, `platform-relay.v1`. Messages carry no version field.
- Adding an optional field is not a new version. Receivers ignore unknown fields.
- Changing a field's meaning, removing one, or changing the message flow is `platform-relay.v2`.
- An agent may offer several versions, newest first: `platform-relay.v2, platform-relay.v1`. The
  relay answers with the one it picked. An agent refused at the upgrade backs off and retries; it
  does not fall back to an unversioned connection.

## 7. Limits

- 16 MiB per frame, both directions (`MAX_FRAME_BYTES`); the relay refuses a request body over
  15 MiB before anything is sent (`MAX_BODY_BYTES`).
- `error.message` is cut at 200 characters (`MAX_MESSAGE_CHARS`): it quotes parts of the peer's
  frame, and the error must fit a frame itself.
- Timers: `register` within 10 s of the upgrade, `heartbeat` every 30 s, stale after 90 s; the
  agent waits `timeout_s` (100) for VS and the relay 110 s for the agent.
