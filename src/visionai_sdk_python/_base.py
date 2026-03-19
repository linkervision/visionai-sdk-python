

class _BaseSdkClient:
    """Base class for VisionAI SDK clients.

    Holds shared connection configuration and provides
    common URL/header builder utilities.
    """

    def __init__(
        self,
        auth_url: str,
        vlm_url: str,
        verify_ssl: bool = True,
        timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        """Initialize the client with connection settings.

        Args:
            auth_url: Base URL for the authentication service.
            vlm_url: Base URL for the VLM inference service.
            verify_ssl: Whether to verify TLS certificates.
            timeout: Default request timeout in seconds.
            max_connections: Maximum number of concurrent connections in the pool.
            max_keepalive_connections: Maximum number of idle keep-alive connections.
        """
        self.auth_url = auth_url
        self.vlm_url = vlm_url
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections


    def _build_url(self, base_url: str, path: str) -> str:
        """Join a base URL and a path, normalizing slashes."""
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


    def _build_header(self, access_token: str | None = None) -> dict[str, str]:
        """Build common HTTP headers, optionally including a Bearer token."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers
