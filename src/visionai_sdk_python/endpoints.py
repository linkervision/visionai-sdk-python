from enum import StrEnum


class AuthEndpoint(StrEnum):
    CLIENT_TOKEN = "/api/users/client-token"
    LOGIN = "/api/users/jwt"


class VLMEndpoint(StrEnum):
    CHAT = "/api/chat"
