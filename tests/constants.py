AUTH_URL = "https://auth.example.com"
VLM_URL = "https://vlm.example.com"

TOKEN_PAYLOAD = {
    "access_token": "test-jwt-token",
    "expires_in": 3600,
    "token_type": "Bearer",
}

VALID_NIM_PAYLOAD = {
    "img": "data:image/jpeg;base64,abc123",
    "prompt": "Describe the image",
}

NORMAL_PENDING_RESPONSE = {"chat_id": "id-001", "status": "pending", "message": None}
NORMAL_RUNNING_RESPONSE = {"chat_id": "id-001", "status": "running", "message": None}
NORMAL_COMPLETED_RESPONSE = {
    "chat_id": "id-001",
    "status": "completed",
    "message": "done",
}
ERROR_FAILED_RESPONSE = {
    "chat_id": "id-001",
    "status": "failed",
    "error": "inference error",
}
ERROR_TIMEOUT_RESPONSE = {
    "chat_id": "id-001",
    "status": "timeout",
    "error": "request timed out",
}
