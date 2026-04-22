import asyncio
import base64
import os

from puremagic import magic_file

from ._base import _BaseClient
from .async_client import AsyncClient
from .client import Client
from .vlm.models import NIMRequestModel

_TEST_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3UmU4c3l4bFRxVmJHeWVoQ000Y2tIMm9rbzVGYVpxTEUtektKeTJlNktZIn0.eyJleHAiOjE3NzQ2MDQ0MDQsImlhdCI6MTc3NDM0NTIwNCwianRpIjoidHJydGNjOmI4ZjA0Y2Y1LWMyMmUtM2JkYi1hYTE3LTNjMmRmODQ2ZjMxNyIsImlzcyI6Imh0dHBzOi8vb2ZmbGluZS52aXNpb25haS5saW5rZXJ2aXNpb24uY29tL2tleWNsb2FrL3JlYWxtcy9saW5rZXItcGxhdGZvcm0iLCJhdWQiOlsicGxhdGZvcm0tYWRtaW4iLCJyZWFsbS1tYW5hZ2VtZW50IiwiYWNjb3VudCJdLCJzdWIiOiIyYzdkZWExZC03YzI3LTRkNDAtOWFjYS1iNDQ2OTE2NTZjMjciLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJwbGF0Zm9ybS1hZG1pbiIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy1saW5rZXItcGxhdGZvcm0iLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsicmVhbG0tbWFuYWdlbWVudCI6eyJyb2xlcyI6WyJtYW5hZ2UtdXNlcnMiLCJ2aWV3LWNsaWVudHMiLCJxdWVyeS1jbGllbnRzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6ImVtYWlsIHByb2ZpbGUiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImNsaWVudEhvc3QiOiI1Mi4zNy4zMy40NSIsInByZWZlcnJlZF91c2VybmFtZSI6InNlcnZpY2UtYWNjb3VudC1wbGF0Zm9ybS1hZG1pbiIsImNsaWVudEFkZHJlc3MiOiI1Mi4zNy4zMy40NSIsImNsaWVudF9pZCI6InBsYXRmb3JtLWFkbWluIn0.OfCZynIMY5-VjrgIAkV3_KgxmVkJmpUYuyFK2WojUsiXTdBv1nZcp45DwqbwNo0ZEBpuPMxmgy38bVCsTy_plCMgsKi3mur3Z8a9wTGiK0x7Iq0rSIXd7cgBdcIhof2tAd1XWHcvjQYl3VX1jik3vyAZftzfq5nMsUzuUoRHopc9utKmAs78j98Pe876laljSifkR9_KLz3pEwrdN4oBMO3W0r6R7yakfTy1m-eznBzNexpeWk98fJWQfkqX72_vR6OP9Z_4LDEe28Z_W6DhqGeXicyBFuCqH-ftw25G3L_CKNvbZZ6BCTCJNAw7H_tCRMUDqTFCz6rDWIMC89xmYQ"


def _encode_image(img_path: str) -> str:
    """
    Reads an image file, determines its MIME type, and encodes it as a Base64 data URI.
    Only JPEG and PNG formats are allowed.
    """
    if not os.path.exists(img_path):
        raise ValueError(f"Error: Image file not found at {img_path}")

    magic_results = magic_file(img_path)
    if not magic_results:
        raise ValueError("Cannot determine the file type of the image")

    mime_type = magic_results[0].mime_type

    if mime_type not in ["image/jpeg", "image/png"]:
        raise ValueError("Only JPEG and PNG images are supported")

    with open(img_path, "rb") as fd:
        img_data = fd.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")

    return f"data:{mime_type};base64,{img_base64}"


async def main() -> None:
    cx = _BaseClient(
        auth_url="https://offline.visionai.linkervision.com",
        vlm_url="https://your-vlm-server.com",
    )
    try:
        claims = await cx._introspect_token_async(_TEST_TOKEN)
        print(claims)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


def enter():
    # Test client credentials flow
    with Client(
        auth_url="https://lighthouse-production.visionai.linkervision.ai",
        vlm_url="http://localhost:8080",
        timeout=30.0,
    ) as client:
        # token = client.auth.login("tonyyang@linkervision.com", "zcg5epm8eku!FWT-bgp")

        token = client.auth.get_access_token(
            "platform-offline-client-api", "platform-lighthouse-client-api-secret"
        )
        # token = client.auth.get_access_token("uoLoPGDqIyB6YccMVcSq37p4zizyyIjO", "3xVUkL4dMVu_BwkNBdMLGE1gm9b6fnHPaAC_osUDubPFeMn4lKpYVf2PXGAtHGx8")
        print(f"Access token: {token.access_token}...")
        print(f"Token expires in: {token.expires_in}s")

        print(f"Token management - access token: {client._access_token}...")
        print(f"Token expires in: {client._token_expires_at}s")

        payload = {
            "img": _encode_image("image_0005.jpg"),
            "prompt": "Are there any car accident in the picture? Is it car or motocycle?",
        }
        response = client.vlm.chat(payload)
        print(response)
        # if valid := client.auth.is_token_valid(token.access_token):
        #     print("Validate valid token...")
        #     print(f"access_token valid")
        # if not (fake := client.auth.is_token_valid(token.access_token[:20])):
        #     print("Validate invalid token...")
        #     print(f"access_token invalid")

        # iss = client._jwt_verifier._get_issuer(token.access_token)
        # print(f"iss _ coo;: {iss}")
        # if not client._jwt_verifier._validate_issuer(iss):
        #     print("True")
        # fake_request = NIMRequestModel(
        #     img=_encode_image("image_0005.jpg"),
        #     prompt="Describe what you see in this image.",
        # )
        # payload = {
        #     "img": _encode_image("image_0005.jpg"),
        #     "prompt": "Are there any car accident in the picture? Is it car or motocycle?",
        # }
        # response = client.chat(
        #     token.access_token,
        #     payload,
        # )
        # print(response)
        # result = client.get_chat(token.access_token, "low:1774935727322-0")
        # print(result)
    # # # Test email/password login
    # with Client(
    #     auth_url="https://offline.visionai.linkervision.com",
    #     vlm_url="https://your-vlm-server.com"
    # ) as client:
    #     token = client.auth.login("tonyyang@linkervision.com", "zcg5epm8eku!FWT-bgp")
    #     print(f"Access token: {token.access_token[:20]}...")
    #     print(f"Token expires in: {token.expires_in}s")

    # client = Client(
    #     auth_url="https://offline.visionai.linkervision.com",
    #     vlm_url="https://your-vlm-server.com"
    # )
    # # Test 401 - Invalid credentials
    # try:
    #     client.auth.get_access_token("user@example.com", "wrong-password")
    # except AuthenticationError as e:
    #     print(f"✓ AuthenticationError: {e}")
    #     print(f"  → status_code: {e.status_code}")
    #     assert e.status_code == 401, "AuthenticationError should have status_code 401"

    # # Test Network errors - Connection timeout/refused
    # try:
    #     broken_client = Client(auth_url="https://invalid-host-12345.example",
    # vlm_url="https://vlm.example.com", timeout=2.0)
    #     broken_client.auth.login("user@example.com", "password")
    # except NetworkError as e:
    #     print(f"✓ NetworkError: {e}")

    # # Test Input validation
    # try:
    #     client.auth.login("", "password")
    # except ValueError as e:
    #     print(f"✓ ValueError (empty email): {e}")

    # try:
    #     client.auth.get_access_token("client-id", "   ")
    # except ValueError as e:
    #     print(f"✓ ValueError (whitespace secret): {e}")

    # # Test consistent status_code access when catching VisionaiSDKError
    # try:
    #     client.get_access_token("invalid-id", "invalid-secret")
    # except ValueError as e:
    #     print(f"✓ ValueError (whitespace invalid-id): {e}")
    # except VisionaiSDKError as e:
    #     if isinstance(e, APIError):
    #         print(f"✓ Caught VisionaiSDKError as APIError with status_code: {e.status_code}")
    #     else:
    #         print(f"✓ Caught VisionaiSDKError (not an APIError): {type(e).__name__}")


async def enter_async():
    print("Testing async......")
    # Test client credentials flow
    async with AsyncClient(
        auth_url="https://offline.visionai.linkervision.com",
        vlm_url="https://your-vlm-server.com",
    ) as client:
        token = await client.auth.get_access_token(
            "platform-admin", "platform-admin-secret"
        )
        print(f"Access token: {token.access_token[:20]}...")
        print(f"Token expires in: {token.expires_in}s")
        if valid := await client.auth.is_token_valid(token.access_token):
            print("Validate valid token...")
            print(f"access_token valid")
        if not (fake := await client.auth.is_token_valid(token.access_token[:20])):
            print("Validate invalid token...")
            print(f"access_token invalid")
    # Test email/password login
    # async with AsyncClient(
    #     auth_url="https://offline.visionai.linkervision.com",
    #     vlm_url="https://your-vlm-server.com"
    # ) as client:
    #     token = await client.auth.login("tonyyang@linkervision.com", "zcg5epm8eku!FWT-bgp")
    #     print(f"Access token: {token.access_token[:20]}...")
    #     print(f"Token expires in: {token.expires_in}s")
    #     if valid := await client.auth.is_token_valid(token.access_token):
    #         print("Validate valid token...")
    #         print(f"access_token valid")
    #     if not (fake := await client.auth.is_token_valid(token.access_token[:20])):
    #         print("Validate invalid token...")
    #         print(f"access_token invalid")

    # client = AsyncClient(
    #     auth_url="https://offline.visionai.linkervision.com",
    #     vlm_url="https://your-vlm-server.com"
    # )
    # # Test 401 - Invalid credentials
    # try:
    #     await client.auth.get_access_token("user@example.com", "wrong-password")
    # except AuthenticationError as e:
    #     print(f"✓ AuthenticationError: {e}")
    #     print(f"status code: {e.status_code}")
    # # Test Network errors - Connection timeout/refused
    # try:
    #     broken_client = AsyncClient(auth_url="https://invalid-host-12345.example",
    # vlm_url="https://vlm.example.com", timeout=2.0)
    #     await broken_client.auth.login("user@example.com", "password")
    # except NetworkError as e:
    #     print(f"✓ NetworkError: {e}")
    # # Test Input validation
    # try:
    #     await client.auth.login("", "password")
    # except ValueError as e:
    #     print(f"✓ ValueError (empty email): {e}")

    # try:
    #     await client.auth.get_access_token("client-id", "   ")
    # except ValueError as e:
    #     print(f"✓ ValueError (whitespace secret): {e}")


# 1x1 red pixel PNG encoded as base64 data URI
# _TEST_IMG_B64 = _encode_image("image_0005.jpg")

# PAYLOAD = NIMRequestModel(
#     img=_TEST_IMG_B64,
#     prompt="Can you summarize this picture?",
#     temperature=0.1,
#     max_tokens=800,
#     top_p=0.1,
#     stream=False,
#     use_cache=False,
#     num_beams=1,
# )


# def chat_enter():
#     c = Client(
#         auth_url="https://lighthouse-production.visionai.linkervision.ai",
#         vlm_url="http://localhost:8080",
#     )
#     try:
#         token = c.auth.get_access_token("observ-client-api", "observ-client-api-secret")
#         print(token)
#     except Exception as e:
#         print(f"{type(e).__name__}: {e}")
#     result = c.vlm.chat(PAYLOAD)
#     print(result)


if __name__ == "__main__":
    # asyncio.run(main())
    enter()
    # asyncio.run(enter_async())
    # chat_enter()
