# Maps known VisionAI server base URLs to their trusted JWT issuers.
# Keys are normalized (no trailing slash).
# Values are exact issuer strings as they appear in the JWT `iss` claim.
#
# Long-term: replace this static table with dynamic discovery from the server's
# trusted-issuers endpoint (e.g. GET {auth_url}/api/v1/auth/trusted-issuers).

_KEYCLOAK_REALM_PATH = "/keycloak/realms/linker-platform"

# On-premise deployments backed by Keycloak (realm: linker-platform).
# Issuer = base_url + _KEYCLOAK_REALM_PATH for all entries.
_KEYCLOAK_BASE_URLS: list[str] = [
    "https://offline.visionai.linkervision.com",
    "https://lighthouse.visionai.linkervision.ai",
    "https://lighthouse-production.visionai.linkervision.ai",
]

# Cloud deployments backed by Auth0 (one tenant per environment).
_AUTH0_URL_TO_ISSUERS: dict[str, list[str]] = {
    "https://visionai.linkervision.com": [
        "https://data-engine-prod.us.auth0.com",
    ],
    "https://staging.visionai.linkervision.com": [
        "https://data-engine-staging.jp.auth0.com",
    ],
    "https://dev2.visionai.linkervision.com": [
        "https://data-engine-dev2.jp.auth0.com",
    ],
    "https://dev.visionai.linkervision.com": [
        "https://dev-045acunea5v1mm3l.us.auth0.com",
    ],
}

_AUTH_URL_TO_ISSUERS: dict[str, list[str]] = {
    **{url: [f"{url}{_KEYCLOAK_REALM_PATH}"] for url in _KEYCLOAK_BASE_URLS},
    **_AUTH0_URL_TO_ISSUERS,
}


def resolve_allowed_issuers(auth_url: str) -> list[str]:
    """Return the trusted issuers for the given auth_url.

    If the URL is in the known table, return its configured issuers.
    Otherwise, assume a Keycloak deployment and derive the issuer from auth_url.
    """
    normalized = auth_url.rstrip("/")
    if known := _AUTH_URL_TO_ISSUERS.get(normalized):
        return known
    return [f"{normalized}{_KEYCLOAK_REALM_PATH}"]
