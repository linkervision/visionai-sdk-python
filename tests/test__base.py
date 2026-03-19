import pytest

from visionai_sdk_python._base import _BaseSdkClient


@pytest.mark.parametrize(
    "base_url, path, expected",
    [
        # Standard case
        ("https://auth.example.com", "token", "https://auth.example.com/token"),
        # Trailing slash on base
        ("https://auth.example.com/", "token", "https://auth.example.com/token"),
        # Leading slash on path
        ("https://auth.example.com", "/token", "https://auth.example.com/token"),
        # Both have slashes — should not double-slash
        ("https://auth.example.com/", "/token", "https://auth.example.com/token"),
        # Nested path
        ("https://vlm.example.com", "v1/infer", "https://vlm.example.com/v1/infer"),
        # Empty path — should not raise, returns base with trailing slash
        ("https://auth.example.com", "", "https://auth.example.com/"),
    ],
)
def test_slash_normalization(base_url: str, path: str, expected: str) -> None:
    assert _BaseSdkClient._build_url(base_url, path) == expected


def test_auth_header() -> None:
    headers = _BaseSdkClient._build_auth_header(access_token="my-token")
    assert headers["Authorization"] == "Bearer my-token"


@pytest.mark.parametrize("empty_url", ["", "  "])
def test_init_raises_on_empty_auth_url(empty_url: str) -> None:
    with pytest.raises(ValueError, match="auth_url must not be empty"):
        _BaseSdkClient(auth_url=empty_url, vlm_url="https://vlm.example.com")


@pytest.mark.parametrize("empty_url", ["", "  "])
def test_init_raises_on_empty_vlm_url(empty_url: str) -> None:
    with pytest.raises(ValueError, match="vlm_url must not be empty"):
        _BaseSdkClient(auth_url="https://auth.example.com", vlm_url=empty_url)


def test_auth_header_raises_on_empty_token() -> None:
    with pytest.raises(ValueError, match="access_token must not be empty"):
        _BaseSdkClient._build_auth_header(access_token="")
