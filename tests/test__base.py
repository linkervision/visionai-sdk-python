import pytest

from visionai_sdk_python._base import _BaseSdkClient


@pytest.fixture
def client() -> _BaseSdkClient:
    return _BaseSdkClient(
        auth_url="https://auth.example.com",
        vlm_url="https://vlm.example.com",
    )


class TestBuildUrl:
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
        ],
    )
    def test_slash_normalization(
        self, client: _BaseSdkClient, base_url: str, path: str, expected: str
    ) -> None:
        assert client._build_url(base_url, path) == expected


class TestBuildHeader:
    def test_without_token_returns_content_type_only(self, client: _BaseSdkClient) -> None:
        headers = client._build_header()
        assert headers == {"Content-Type": "application/json"}

    def test_with_token_includes_bearer(self, client: _BaseSdkClient) -> None:
        headers = client._build_header(access_token="my-token")
        assert headers["Authorization"] == "Bearer my-token"
        assert headers["Content-Type"] == "application/json"

