# VisionAI SDK for Python

Python client library for VisionAI authentication and Vision Language Model (VLM) inference services.

## Features

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
token = client.login("user@example.com", "your-password")

# Submit VLM inference request
response = client.chat({
    "img": "https://example.com/image.jpg",  # or ["url1", "url2"] for multiple images
    "prompt": "Describe this image",
    "temperature": 0.7,
    "max_tokens": 500
})

print(f"Chat ID: {response.chat_id}")
print(f"Status: {response.status}")

# Poll for results
result = client.get_chat(response.chat_id)
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
        await client.get_access_token(
            client_id="your-client-id",
            client_secret="your-client-secret"
        )

        # Submit inference
        response = await client.chat({
            "img": "https://example.com/image.jpg",
            "prompt": "What objects are in this image?",
            "temperature": 0.2
        })

        # Poll until completed
        while True:
            result = await client.get_chat(response.chat_id)
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
token = client.login("user@example.com", "password")
```

### OAuth Client Credentials

```python
client = Client(auth_url="...", vlm_url="...")
token = client.get_access_token(
    client_id="your-client-id",
    client_secret="your-client-secret"
)
```

Tokens are stored internally and automatically refreshed before expiration.

## VLM Inference

### Submit Chat Request

```python
from visionai_sdk_python.models import NIMRequestModel

# Using dict
response = client.chat({
    "img": "https://example.com/image.jpg",
    "prompt": "Analyze this image",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9
})

# Using typed model
request = NIMRequestModel(
    img=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
    prompt="Compare these images",
    temperature=0.5,
    max_tokens=500
)
response = client.chat(request)
```

### Check Result

```python
result = client.get_chat(response.chat_id)

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
is_valid = client.is_token_valid("eyJhbGci...")
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
    client.login("user@example.com", "wrong-password")
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
    client.login("user@example.com", "password")
    response = client.chat({"img": "...", "prompt": "..."})
# client.close() called automatically

# Async
async with AsyncClient(auth_url="...", vlm_url="...") as client:
    await client.login("user@example.com", "password")
    response = await client.chat({"img": "...", "prompt": "..."})
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

## Requirements

- Python >= 3.11
- httpx >= 0.28.1
- pydantic >= 2.12.5
- cryptography >= 46.0.5
- PyJWT[cryptography] >= 2.8.0


## Support

For issues and questions, please open an issue on GitHub.
