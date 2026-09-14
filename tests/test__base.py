import jwt
import pytest

from visionai_sdk_python._base import _BaseClient
from visionai_sdk_python.constants import resolve_allowed_issuers


def test_init_stores_attributes() -> None:
    client = _BaseClient(
        auth_url="https://auth.example.com", vlm_url="https://vlm.example.com"
    )
    assert client.auth_url == "https://auth.example.com"
    assert client.vlm_url == "https://vlm.example.com"
    assert client.verify_ssl is True
    assert client.timeout == 10.0
    assert client.max_connections == 100
    assert client.max_keepalive_connections == 20


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
    assert _BaseClient._build_url(base_url, path) == expected


def test_auth_header() -> None:
    headers = _BaseClient._build_auth_header(access_token="my-token")
    assert headers["Authorization"] == "Bearer my-token"


@pytest.mark.parametrize("empty_url", ["", "  "])
def test_init_raises_on_empty_auth_url(empty_url: str) -> None:
    with pytest.raises(ValueError, match="auth_url"):
        _BaseClient(auth_url=empty_url, vlm_url="https://vlm.example.com")


@pytest.mark.parametrize("empty_url", ["", "  "])
def test_init_raises_on_empty_vlm_url(empty_url: str) -> None:
    with pytest.raises(ValueError, match="vlm_url"):
        _BaseClient(auth_url="https://auth.example.com", vlm_url=empty_url)


def test_auth_header_raises_on_empty_token() -> None:
    with pytest.raises(ValueError, match="access_token"):
        _BaseClient._build_auth_header(access_token="")


_KEYCLOAK_SITES = [
    "https://staging.visionai.linkervision.com",
    "https://dev.visionai.linkervision.com",
    "https://dev2.visionai.linkervision.com",
    "https://dev3.visionai.linkervision.com",
    "https://offline.visionai.linkervision.com",
    "https://lighthouse.visionai.linkervision.ai",
    "https://lighthouse-production.visionai.linkervision.ai",
]


@pytest.mark.parametrize("auth_url", _KEYCLOAK_SITES)
def test_resolve_allowed_issuers_for_keycloak_sites(auth_url: str) -> None:
    """Every site that has cut over to Keycloak resolves to its realm issuer."""
    assert resolve_allowed_issuers(auth_url) == [
        f"{auth_url}/keycloak/realms/linker-platform"
    ]


def test_resolve_allowed_issuers_for_auth0_site() -> None:
    """Production is the last Auth0 site and keeps its tenant issuer."""
    assert resolve_allowed_issuers("https://visionai.linkervision.com") == [
        "https://data-engine-prod.us.auth0.com"
    ]


def test_resolve_allowed_issuers_falls_back_to_keycloak() -> None:
    """An unknown host is assumed to be a Keycloak deployment."""
    assert resolve_allowed_issuers("https://somewhere.example.com") == [
        "https://somewhere.example.com/keycloak/realms/linker-platform"
    ]


@pytest.mark.parametrize(
    "auth_url",
    [
        "https://dev.visionai.linkervision.com/",
        "https://dev.visionai.linkervision.com///",
    ],
)
def test_resolve_allowed_issuers_ignores_trailing_slashes(auth_url: str) -> None:
    assert resolve_allowed_issuers(auth_url) == [
        "https://dev.visionai.linkervision.com/keycloak/realms/linker-platform"
    ]


def test_client_defaults_to_resolved_issuers() -> None:
    """Omitting allowed_issuers pins the client to the site's own issuer."""
    client = _BaseClient(
        auth_url="https://dev.visionai.linkervision.com",
        vlm_url="https://vlm.example.com",
    )
    client._jwt_verifier._validate_issuer(
        "https://dev.visionai.linkervision.com/keycloak/realms/linker-platform"
    )
    with pytest.raises(jwt.InvalidIssuerError):
        client._jwt_verifier._validate_issuer("https://data-engine-dev2.jp.auth0.com")


_ALLOWED_ISSUERS = [
    "https://offline.visionai.linkervision.com/keycloak/realms/linker-platform",
    "https://data-engine-prod.us.auth0.com",
]


@pytest.mark.parametrize("issuer", _ALLOWED_ISSUERS)
def test_validate_issuer_passes_for_allowed(issuer: str) -> None:
    client = _BaseClient(
        auth_url="https://auth.example.com",
        vlm_url="https://vlm.example.com",
        allowed_issuers=_ALLOWED_ISSUERS,
    )
    client._jwt_verifier._validate_issuer(issuer)  # should not raise


@pytest.mark.parametrize(
    "issuer",
    [
        "https://evil.com/keycloak/realms/linker-platform",
        "https://attacker.auth0.com/",
        "https://offline.visionai.linkervision.com/keycloak/realms/other-realm",
    ],
)
def test_validate_issuer_raises_for_unknown(issuer: str) -> None:
    client = _BaseClient(
        auth_url="https://auth.example.com",
        vlm_url="https://vlm.example.com",
        allowed_issuers=_ALLOWED_ISSUERS,
    )
    with pytest.raises(jwt.InvalidIssuerError):
        client._jwt_verifier._validate_issuer(issuer)


def test_validate_issuer_raises_for_unknown_when_no_allowed_issuers() -> None:
    client = _BaseClient(
        auth_url="https://auth.example.com",
        vlm_url="https://vlm.example.com",
    )
    with pytest.raises(jwt.InvalidIssuerError):
        client._jwt_verifier._validate_issuer("https://anyone.com/")  # should not raise
