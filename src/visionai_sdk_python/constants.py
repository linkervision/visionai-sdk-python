# Resolves the trusted JWT issuers for a VisionAI server base URL.
#
# Every deployment hosts its own Keycloak behind the site's public base URL, so
# the realm issuer is derived from `auth_url` rather than kept in a table. Only
# sites that additionally accept tokens from an external IdP need an entry here.
#
# Long-term: replace this with dynamic discovery from the server's
# trusted-issuers endpoint (e.g. GET {auth_url}/api/v1/auth/trusted-issuers).

_KEYCLOAK_REALM_PATH = "/keycloak/realms/linker-platform"

# Sites whose tokens may still come from an external IdP, in addition to their
# own Keycloak realm. Production has not cut over yet; listing both issuers lets
# an installed SDK keep working across the migration without an upgrade.
#
# Drop the entry once production is confirmed migrated — a retired Auth0 tenant
# name can be re-registered by someone else, so it should not stay trusted
# indefinitely.
_EXTERNAL_IDP_ISSUERS: dict[str, list[str]] = {
    "https://visionai.linkervision.com": [
        "https://data-engine-prod.us.auth0.com",
    ],
}


def resolve_allowed_issuers(auth_url: str) -> list[str]:
    """Return the trusted issuers for the given auth_url.

    Always includes the site's own Keycloak realm issuer, plus any external IdP
    issuer the site still accepts.
    """
    normalized = auth_url.rstrip("/")
    return [
        f"{normalized}{_KEYCLOAK_REALM_PATH}",
        *_EXTERNAL_IDP_ISSUERS.get(normalized, []),
    ]
