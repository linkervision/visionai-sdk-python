# VisionAI SDK for Python

Python client library for VisionAI authentication and Vision Language Model (VLM) inference services.

## Features

- **Modular Architecture**: Feature-based organization (auth, vlm) for easy extension
- **Dual Authentication**: Email/password login or OAuth client credentials
- **Auto Token Management**: Automatic token refresh before expiration
- **JWT Validation**: Built-in token signature and expiration verification
- **VLM Inference**: Submit and poll vision-language model tasks
- **Resize Planning**: Compute VLM input dimensions (smart resize / square / longest edge) without any image dependency — the caller does the actual resize
- **Async Support**: Full async/await support with `AsyncClient`
- **Type Safe**: Full type hints with Pydantic validation
- **Service Source Attribution**: Auto-attach a service-source header to every outbound `requests`/`httpx`/`aiohttp` call with a single startup call

## Installation

```bash
pip install visionai-sdk-python
```

Attributing direct `requests`/`aiohttp` calls, or using `instrumentation.instrument()` at all (see [Service Source Attribution](#service-source-attribution) below), needs optional extras:

```bash
pip install visionai-sdk-python[requests]
pip install visionai-sdk-python[aiohttp]
pip install visionai-sdk-python[instrumentation]   # needed for instrument() itself
```

`httpx` is already a core dependency, so it needs no extra. `Client`/`AsyncClient` attribute themselves without any of the above.

## Quick Start

### Synchronous Usage

```python
from visionai_sdk_python import Client

# Initialize client
client = Client(
    auth_url="https://auth.visionai.example.com",
    vlm_url="https://vlm.visionai.example.com"
)

# Login with email/password
token = client.auth.login("user@example.com", "your-password")

# Submit VLM inference request
response = client.vlm.chat({
    "img": "examplebase64",
    "prompt": "Describe this image",
    "temperature": 0.7,
    "max_tokens": 500
})

print(f"Chat ID: {response.chat_id}")
print(f"Status: {response.status}")

# Poll for results
result = client.vlm.get_chat(response.chat_id)
if result.status == "completed":
    print(f"Result: {result.message}")

# Close client when done
client.close()
```

### Asynchronous Usage

```python
import asyncio
from visionai_sdk_python import AsyncClient

async def main():
    async with AsyncClient(
        auth_url="https://auth.visionai.example.com",
        vlm_url="https://vlm.visionai.example.com"
    ) as client:
        # OAuth client credentials flow
        await client.auth.get_access_token(
            client_id="your-client-id",
            client_secret="your-client-secret"
        )

        # Submit inference
        response = await client.vlm.chat({
            "img": "examplebase64",
            "prompt": "What objects are in this image?",
            "temperature": 0.2
        })

        # Poll until completed
        while True:
            result = await client.vlm.get_chat(response.chat_id)
            if result.status in ("completed", "failed", "timeout"):
                break
            await asyncio.sleep(1)

        if result.status == "completed":
            print(f"Answer: {result.message}")
        else:
            print(f"Error: {result.error}")

asyncio.run(main())
```

## Authentication

### Email/Password Login

```python
client = Client(auth_url="...", vlm_url="...")
token = client.auth.login("user@example.com", "password")
```

### OAuth Client Credentials

```python
client = Client(auth_url="...", vlm_url="...")
token = client.auth.get_access_token(
    client_id="your-client-id",
    client_secret="your-client-secret"
)
```

Tokens are stored internally and automatically refreshed before expiration.

## VLM Inference

### Submit Chat Request

```python
from visionai_sdk_python.vlm.models import NIMRequestModel

# Using dict
response = client.vlm.chat({
    "img": "examplebase64",
    "prompt": "Analyze this image",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9
})

# Using typed model
request = NIMRequestModel(
    img=["examplebase64"],
    prompt="Compare these images",
    temperature=0.5,
    max_tokens=500
)
response = client.vlm.chat(request)
```

### Check Result

```python
result = client.vlm.get_chat(response.chat_id)

if result.status == "completed":
    print(result.message)
elif result.status in ("failed", "timeout"):
    print(f"Error: {result.error}")
```

**Response Status:**
- `pending`: Request queued
- `running`: Processing
- `completed`: Success, check `message`
- `failed`: Error, check `error`
- `timeout`: Request timeout

### Resize Planning

Compute the target dimensions for a VLM input image. Pure math, no image
dependencies — resize with whatever library your service already uses,
applying `plan.resampling`:

```python
from visionai_sdk_python.vlm import RESIZE_OPTIONS, compute_resize

# From a shared UI option (single source of truth — do not copy this table)
plan = compute_resize(1080, 1920, **RESIZE_OPTIONS["smart_768_p32"])
# ResizePlan(width=1024, height=576, resampling='bilinear')

# Or explicit modes (pick exactly one)
compute_resize(h, w, factor=32, max_pixels=768 * 768)  # smart: patch-aligned, pixel-capped
compute_resize(h, w, square=384)                       # exact n x n
compute_resize(h, w, longest_edge=768)                 # cap longer side, never upscales

# Apply with your own imaging library (PIL shown; see module docstring for cv2)
resampling = {"bilinear": Image.Resampling.BILINEAR,
              "bicubic": Image.Resampling.BICUBIC,
              "lanczos": Image.Resampling.LANCZOS}[plan.resampling]
img = img.resize((plan.width, plan.height), resampling)
```

Invalid input (non-positive dimensions or targets, conflicting modes,
`min_pixels > max_pixels`, an unsatisfiable pixel budget, unknown
`resampling`) raises `ValueError`.

## Token Validation

```python
# Validate any JWT token
is_valid = client.auth.is_token_valid("eyJhbGci...")
if is_valid:
    print("Token is valid")
else:
    print("Token is expired or invalid")
```

## Configuration

```python
client = Client(
    auth_url="https://auth.example.com",
    vlm_url="https://vlm.example.com",
    allowed_issuers=["https://auth.example.com"],  # Optional: restrict token issuers
    verify_ssl=True,                                # SSL verification
    timeout=10.0,                                   # Request timeout in seconds
    max_connections=100,                            # Connection pool size
    max_keepalive_connections=20                    # Keepalive connections
)
```

## Error Handling

```python
from visionai_sdk_python import (
    VisionaiSDKError,
    AuthenticationError,
    NetworkError,
    ClientError,
    ServerError
)

try:
    client.auth.login("user@example.com", "wrong-password")
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except NetworkError as e:
    print(f"Network error: {e}")
except VisionaiSDKError as e:
    print(f"SDK error: {e}")
```

**Exception Hierarchy:**
- `VisionaiSDKError`: Base exception
  - `AuthenticationError`: 401 Unauthorized
  - `PermissionDeniedError`: 403 Forbidden
  - `ClientError`: 4xx client errors
  - `ServerError`: 5xx server errors
  - `NetworkError`: Connection/timeout errors
  - `JwksDiscoveryError`: OIDC discovery failures

## Context Manager Usage

Recommended for automatic resource cleanup:

```python
# Sync
with Client(auth_url="...", vlm_url="...") as client:
    client.auth.login("user@example.com", "password")
    response = client.vlm.chat({"img": "...", "prompt": "..."})
# client.close() called automatically

# Async
async with AsyncClient(auth_url="...", vlm_url="...") as client:
    await client.auth.login("user@example.com", "password")
    response = await client.vlm.chat({"img": "...", "prompt": "..."})
# client.close() called automatically
```

## Service Source Attribution

Outbound requests to VisionAI carry an `X-Request-Source` header naming the
service that made the call, so downstream systems can attribute VLM token usage
back to it. Set `VISIONAI_SERVICE_SOURCE` to the deployment's stable service name
(e.g. via the Kubernetes Downward API, reading the pod's `app` label).

### Setting this up for your service

If you're adopting this from another service (not the SDK's own maintainers),
here's the checklist:

1. Set `VISIONAI_SERVICE_SOURCE` in your deployment to your own stable service
   name — this is the value that ends up on the wire, so make it something a
   human reading a Prometheus label or a token-usage report would recognize.
2. If you use `Client`/`AsyncClient`, you're done — skip to the next section.
   If you call `requests`/`httpx`/`aiohttp` directly, call `instrument()` once
   at startup **with the list of hosts your service actually calls out to**:

   ```python
   from visionai_sdk_python import instrumentation

   instrumentation.instrument(
       allowed_destination_hosts=["vlm-inference-server", "vlm-scheduling-service"],
   )
   ```

3. **There is no default — `allowed_destination_hosts` is required on the
   first call.** Omitting it raises `ValueError`. This is deliberate: a
   built-in default that happened not to match your service's real hosts
   would let `instrument()` run without error while `X-Request-Source`
   silently never went out anywhere, which looks identical to "it's not
   working" with no error to point at. `instrumentation.
   SUGGESTED_ALLOWED_DESTINATION_HOSTS` is a reasonable starting point
   (`vlm-inference-server`, `vlm-scheduling-service`, `*.svc.cluster.local`,
   ...) if you want to pass it explicitly rather than typing out your own list.
4. If in doubt about what your service actually calls, check the host in the
   URLs you pass to `requests`/`httpx`/`aiohttp` today — that's exactly what
   `allowed_destination_hosts` needs to match.

The rest of this section covers what `instrument()` does and why, in more
detail.

### Using `Client` / `AsyncClient`

Nothing to do. The SDK's own clients pick the variable up automatically, fresh
on every request — including a later `current_origin()` scope (A-5), with no
manual header merging needed. If the service also calls `instrument()`, the
client's own injection cooperates with it correctly (same destination scoping
applies to the client's own calls too, not just direct `requests`/`httpx`/
`aiohttp` use).

### Using `requests` / `httpx` / `aiohttp` directly

Call `instrument()` once at service startup. **No other code changes** — your
existing `import requests` and every call site stay exactly as they are.

```python
from visionai_sdk_python import instrumentation

instrumentation.instrument(
    allowed_destination_hosts=["vlm-inference-server", "*.svc.cluster.local"],
)
```

This wraps a request-dispatch method of `requests.Session`, `httpx.Client`/
`AsyncClient` and `aiohttp.ClientSession` in place, so it also covers calls made
by third-party packages you cannot edit, module-level one-shots like
`requests.get()`, and requests sent through a client instance that already
existed before `instrument()` ran (see "Call it any time" below).

`instrument()` itself needs the `instrumentation` extra (`wrapt`). The libraries
it patches are separately optional, and whichever is not installed is skipped:

```bash
pip install visionai-sdk-python[instrumentation]
pip install visionai-sdk-python[requests]   # httpx is already a core dependency
pip install visionai-sdk-python[aiohttp]
```

### Destination scoping

Only destinations matching `allowed_destination_hosts` receive the header;
every other destination — Stripe, an external webhook, anything outside the
allowlist — is completely unaffected, on every request including ones made by
third-party code. This is **fail-closed**: there is no "match everything" mode,
so a service opts its own internal hosts in explicitly rather than getting them
for free. Patterns are matched against the parsed URL's hostname (case
insensitive, `fnmatch`-style globs, e.g. `*.svc.cluster.local`), never against
the raw URL string, so a host can't be spoofed via path or query string.

`allowed_destination_hosts` is required on the first call — see the checklist
above. If, over the life of the process, no destination ever matches the
allowlist, a `RuntimeWarning` fires at interpreter shutdown: a wrong allowlist
and "attribution has no data for some other reason" would otherwise be
indistinguishable in production.

The check re-runs on every hop, including redirects: a request to an
allowlisted host that gets redirected somewhere else stops carrying the header
at that point rather than leaking it to wherever the redirect points. A header
your own code set explicitly is never touched, on any hop or destination.

`source_headers()` remains available as an escape hatch for the one case
per-request scoping can't cover: deliberately sending the header to a
destination outside `allowed_destination_hosts`. See "If your service is called
by another instrumented service" below for how to use it.

### Call it any time

Injection runs on `Session.send`, `Client`/`AsyncClient._send_single_request`
and `aiohttp.ClientRequest.__init__` — a request-dispatch method looked up on
the class at call time, not at construction. So, unlike a `__init__`-time
patch, a client instance built before `instrument()` runs is not a gap: every
call it makes afterwards looks up the same, now-patched method. Calling
`instrument()` early (before anything constructs a client) is still good
practice, since it's one less thing to reason about, but there's no silent
partial-failure mode tied to import order the way there would be if injection
happened at construction time.

`uninstrument()` reverses everything, which is mainly useful for test isolation.

### If your service is called by another instrumented service

By default `instrument()` stamps *your own* `VISIONAI_SERVICE_SOURCE` on every
outbound call. That's correct as long as your service is the true origin of the
VLM calls it makes.

If service A calls your service, and your service then calls VLM as part of
handling A's request, you should forward A's identity rather than stamp your
own — otherwise it's silently lost at that hop, the same problem
`visionai-vlm-scheduling-service` solves for its Redis queue hop by explicitly
storing and re-attaching the header. From your own inbound-request middleware:

```python
from visionai_sdk_python import instrumentation

origin = instrumentation.origin_from_headers(incoming_request.headers)
with instrumentation.current_origin(origin):
    ...  # handle the request, including any outbound VLM calls
```

`origin_from_headers()` reads an inbound `X-Request-Source` case-insensitively
(a mapping or a list of `(key, value)` pairs — whatever your framework's headers
object gives you), returning `None` if the caller didn't set one, which means
*you* are the origin. `current_origin()` is backed by `contextvars`, so it's
isolated per request under concurrency — safe with `asyncio` tasks and threaded
workers alike, and nesting restores the outer value on exit.

This makes forwarding automatic for every request made in that scope, including
through a client built once at startup and reused across many requests —
injection runs right before each request is dispatched rather than when the
client was built, so it reads `current_origin()` fresh every time, not just
once at construction.

The one case that still needs an explicit pass-through is sending the header to
a destination outside `allowed_destination_hosts` on purpose:

```python
response = shared_client.get(
    url, headers={**instrumentation.source_headers(), **other_headers}
)
```

Per-call headers already override a client's defaults in requests/httpx/aiohttp,
so no extra mechanism is needed for that case. `source_headers()` returns
`{"X-Request-Source": value}` (inherited origin, falling back to your own
identity) or `{}` if there is nothing to send. **Do not** pass
`get_current_origin()` directly as a header value — it returns `None` whenever
there is no inherited origin (the common case), and `httpx` raises `TypeError`
on a `None`-valued header.

### Behavior

- **`X-Request-Source` means last hop, not origin, by default.** Without A-5
  set up, it names whichever service called *this* service directly — not
  the request's ultimate origin. A service that wants origin tracking across
  hops opts in explicitly via `current_origin()`/`origin_from_headers()` (see
  "If your service is called by another instrumented service" above); nothing
  here infers origin automatically. This is a deliberate scope decision, not
  a limitation to work around: zero-config attribution ("who called me") is
  enough for most consumers, and origin tracking is opt-in because it needs a
  service to actually wire up inbound middleware, which not every consumer
  will do.
- **Division of labor with `User-Agent`:** this SDK doesn't set `User-Agent`
  at all. If your service's own UA convention already names the caller +
  version for hop-by-hop debugging/logging, that's a separate, orthogonal
  concern from `X-Request-Source` — UA answers "who's on the other end of
  this specific hop", `X-Request-Source` answers "who should this be
  attributed to" (last hop by default, origin if A-5 is set up). Don't rely
  on UA for attribution: it's compound and free-form, not designed to be
  parsed into a stable label, and proxies may rewrite it.
- If `VISIONAI_SERVICE_SOURCE` is unset, no header is added — fully backward
  compatible.
- Only destinations matching `allowed_destination_hosts` get the header — fail
  closed, checked again on every redirect hop, never a "match everything" mode.
- If your own code already sets `X-Request-Source`, that value is respected and
  never overwritten (checked case-insensitively), on any hop or destination.
- The real library classes are wrapped in place, never replaced or subclassed, so
  `isinstance` checks, exception identity and the full public API are unchanged.
  Exceptions are never caught, translated or wrapped.
- A malformed `VISIONAI_SERVICE_SOURCE` is dropped rather than allowed to break
  the request. Surrounding whitespace is stripped, so a trailing newline from a
  Helm block scalar is handled; a value that is still unusable as a header
  (control characters, non-ASCII) is skipped with a `RuntimeWarning` and the
  request proceeds unattributed.

## Development

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

## Extending the SDK

The SDK follows a modular architecture that makes it easy to add new features. Each feature (like `auth` or `vlm`) is organized as a separate module.

### Architecture Overview

```
src/visionai_sdk_python/
├── {feature}/
│   ├── __init__.py          # Feature exports
│   ├── models.py            # Pydantic models for requests/responses
│   ├── _mixin.py            # Shared business logic (no I/O)
│   ├── resource.py          # Sync operations (I/O)
│   └── async_resource.py    # Async operations (I/O)
├── client.py                # Sync client with feature registration
└── async_client.py          # Async client with feature registration
```

### Adding a New Feature

Follow these steps to add a new feature (e.g., `dataset`):

#### 1. Create Feature Directory

```bash
mkdir -p src/visionai_sdk_python/dataset
touch src/visionai_sdk_python/dataset/{__init__.py,models.py,_mixin.py,resource.py,async_resource.py}
```

#### 2. Define Models (`dataset/models.py`)

```python
from pydantic import BaseModel

class Dataset(BaseModel):
    """Dataset response model."""
    id: str
    name: str
    created_at: str

class CreateDatasetRequest(BaseModel):
    """Create dataset request model."""
    name: str
    description: str | None = None
```

#### 3. Implement Shared Logic (`dataset/_mixin.py`)

```python
from .models import CreateDatasetRequest, Dataset

class DatasetMixin:
    """Shared dataset logic (validation, data preparation, parsing).

    This mixin contains all business logic that doesn't involve I/O operations.
    Sync and async resources inherit from this to avoid code duplication.
    """

    def _prepare_create_request(self, payload: CreateDatasetRequest | dict) -> dict:
        """Prepare create dataset request payload."""
        request = (
            CreateDatasetRequest.model_validate(payload)
            if isinstance(payload, dict)
            else payload
        )
        return request.model_dump(mode="json")

    def _parse_dataset_response(self, data: dict) -> Dataset:
        """Parse dataset response from API."""
        return Dataset(**data)
```

#### 4. Implement Sync Resource (`dataset/resource.py`)

```python
from typing import TYPE_CHECKING

from ..endpoints import DatasetEndpoint  # Add to endpoints.py
from .models import CreateDatasetRequest, Dataset
from ._mixin import DatasetMixin

if TYPE_CHECKING:
    from ..client import Client


class DatasetResource(DatasetMixin):
    """Synchronous dataset operations."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    def create(self, payload: CreateDatasetRequest | dict) -> Dataset:
        """Create a new dataset."""
        # Ensure token is valid
        self._client._ensure_token()

        # Prepare request (from Mixin)
        body = self._prepare_create_request(payload)

        # I/O operation (sync)
        response = self._client._request(
            "POST",
            self._client._build_url(self._client.dataset_url, DatasetEndpoint.CREATE),
            headers=self._client._build_auth_header(self._client._access_token),
            json=body,
        )

        # Parse response (from Mixin)
        return self._parse_dataset_response(response.json())
```

#### 5. Implement Async Resource (`dataset/async_resource.py`)

```python
from typing import TYPE_CHECKING

from ..endpoints import DatasetEndpoint
from .models import CreateDatasetRequest, Dataset
from ._mixin import DatasetMixin

if TYPE_CHECKING:
    from ..async_client import AsyncClient


class AsyncDatasetResource(DatasetMixin):
    """Asynchronous dataset operations."""

    def __init__(self, client: "AsyncClient") -> None:
        self._client = client

    async def create(self, payload: CreateDatasetRequest | dict) -> Dataset:
        """Create a new dataset."""
        # Ensure token is valid
        await self._client._ensure_token()

        # Prepare request (from Mixin - same as sync)
        body = self._prepare_create_request(payload)

        # I/O operation (async - only difference)
        response = await self._client._request(
            "POST",
            self._client._build_url(self._client.dataset_url, DatasetEndpoint.CREATE),
            headers=self._client._build_auth_header(self._client._access_token),
            json=body,
        )

        # Parse response (from Mixin - same as sync)
        return self._parse_dataset_response(response.json())
```

#### 6. Export from Feature Module (`dataset/__init__.py`)

```python
from .async_resource import AsyncDatasetResource
from .models import CreateDatasetRequest, Dataset
from .resource import DatasetResource

__all__ = [
    "DatasetResource",
    "AsyncDatasetResource",
    "Dataset",
    "CreateDatasetRequest",
]
```

#### 7. Add Endpoints (`endpoints.py`)

```python
class DatasetEndpoint:
    """Dataset API endpoints."""
    CREATE = "/api/datasets"
    GET = "/api/datasets/{id}"
    LIST = "/api/datasets"
```

#### 8. Register in Clients

If your feature requires a new service URL, **first add it to `_BaseClient`** in `_base.py` — all URL fields are owned by the base class and the subclass merely forwards them via `super().__init__()`.

**`_base.py` (only if adding a new URL):**
```python
class _BaseClient:
    def __init__(
        self,
        auth_url: str,
        vlm_url: str,
        dataset_url: str,           # add new URL parameter
        allowed_issuers: list[str] | None = None,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        ...
        if not dataset_url.strip():
            raise ValueError("dataset_url must not be empty")
        self.dataset_url = dataset_url.strip()   # store it here
        ...
```

**`client.py`:**
```python
from .dataset.resource import DatasetResource

class Client(_BaseClient):
    def __init__(self, auth_url: str, vlm_url: str, dataset_url: str, ...):
        super().__init__(auth_url=auth_url, vlm_url=vlm_url, dataset_url=dataset_url, ...)
        self.auth = AuthResource(self)
        self.vlm = VLMResource(self)
        self.dataset = DatasetResource(self)  # Register new feature
```

**`async_client.py`:**
```python
from .dataset.async_resource import AsyncDatasetResource

class AsyncClient(_BaseClient):
    def __init__(self, auth_url: str, vlm_url: str, dataset_url: str, ...):
        super().__init__(auth_url=auth_url, vlm_url=vlm_url, dataset_url=dataset_url, ...)
        self.auth = AsyncAuthResource(self)
        self.vlm = AsyncVLMResource(self)
        self.dataset = AsyncDatasetResource(self)  # Register new feature
```

#### 9. Usage

```python
from visionai_sdk_python import Client

client = Client(
    auth_url="...",
    vlm_url="...",
    dataset_url="..."
)

# Authenticate
client.auth.login("user@example.com", "password")

# Use new feature
dataset = client.dataset.create({
    "name": "My Dataset",
    "description": "Example dataset"
})
print(f"Created dataset: {dataset.id}")
```

### Key Principles

1. **Mixin Pattern**: All business logic goes in `_mixin.py` to avoid duplication
2. **I/O Separation**: Resources only handle I/O operations (sync vs async)
3. **Type Safety**: Use Pydantic models for validation and type hints
4. **Consistent Structure**: Follow the same folder structure for all features
5. **Client Registration**: Register resources in both `Client` and `AsyncClient`

### Testing New Features

Follow the existing test patterns:

```python
# tests/test_dataset.py
def test_create_dataset_success(mock_client: Client):
    response = mock_client.dataset.create({
        "name": "Test Dataset",
        "description": "Test"
    })
    assert isinstance(response, Dataset)
    assert response.name == "Test Dataset"
```

## Requirements

- Python >= 3.11
- httpx >= 0.28.1
- pydantic >= 2.12.5
- cryptography >= 46.0.5
- PyJWT[cryptography] >= 2.8.0


## Support

For issues and questions, please open an issue on GitHub.
