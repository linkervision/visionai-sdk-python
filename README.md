# VisionAI SDK for Python

Python client library for VisionAI authentication and Vision Language Model (VLM) inference services.

## Features

- **Modular Architecture**: Feature-based organization (auth, vlm) for easy extension
- **Dual Authentication**: Email/password login or OAuth client credentials
- **Auto Token Management**: Automatic token refresh before expiration
- **JWT Validation**: Built-in token signature and expiration verification
- **VLM Inference**: Submit and poll vision-language model tasks
- **Async Support**: Full async/await support with `AsyncClient`
- **Type Safe**: Full type hints with Pydantic validation

## Installation

```bash
pip install visionai-sdk-python
```

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

**`client.py`:**
```python
from .dataset.resource import DatasetResource

class Client(_BaseClient):
    def __init__(self, auth_url: str, vlm_url: str, dataset_url: str, ...):
        super().__init__(...)
        self.auth = AuthResource(self)
        self.vlm = VLMResource(self)
        self.dataset = DatasetResource(self)  # Register new feature
```

**`async_client.py`:**
```python
from .dataset.async_resource import AsyncDatasetResource

class AsyncClient(_BaseClient):
    def __init__(self, auth_url: str, vlm_url: str, dataset_url: str, ...):
        super().__init__(...)
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
