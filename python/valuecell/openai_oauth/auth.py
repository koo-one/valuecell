"""Interactive ChatGPT OAuth flow helpers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

import requests

from valuecell.openai_oauth.exceptions import (
    NotAuthenticatedError,
    OAuthFlowError,
    TokenRefreshError,
)
from valuecell.openai_oauth.store import AuthStore, OAuthCredentials

AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
SCOPE = "openid profile email offline_access"
REDIRECT_URI = "http://localhost:1455/auth/callback"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def create_state() -> str:
    return secrets.token_hex(16)


def build_authorize_url(*, state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(payload + padding)
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return None

    return obj if isinstance(obj, dict) else None


def extract_chatgpt_account_id(payload: dict[str, Any]) -> str | None:
    claim = payload.get("https://api.openai.com/auth")
    if not isinstance(claim, dict):
        return None

    account_id = claim.get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        return None

    return account_id


@dataclass(frozen=True)
class TokenResponse:
    access: str
    refresh: str
    expires_at_ms: int


@dataclass(frozen=True)
class OAuthStatus:
    authenticated: bool
    expires_at_ms: int | None = None
    account_id: str | None = None
    expired: bool = False


def _post_token(
    payload: dict[str, str],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    http = session or requests.Session()
    close_after = session is None
    try:
        response = http.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise OAuthFlowError("OAuth token exchange failed.") from exc
    finally:
        if close_after:
            http.close()

    return data if isinstance(data, dict) else {}


def exchange_authorization_code(
    *,
    code: str,
    verifier: str,
    session: requests.Session | None = None,
) -> TokenResponse:
    data = _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        },
        session=session,
    )

    access = str(data.get("access_token") or "")
    refresh = str(data.get("refresh_token") or "")
    expires_in = int(data.get("expires_in") or 0)
    if not (access and refresh and expires_in):
        raise OAuthFlowError("Token response missing required fields.")

    return TokenResponse(
        access=access,
        refresh=refresh,
        expires_at_ms=int(time.time() * 1000) + expires_in * 1000,
    )


def refresh_access_token(
    *,
    refresh_token: str,
    session: requests.Session | None = None,
) -> TokenResponse:
    http = session or requests.Session()
    close_after = session is None
    try:
        response = http.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise TokenRefreshError("Token refresh failed.") from exc
    finally:
        if close_after:
            http.close()

    if not isinstance(data, dict):
        raise TokenRefreshError("Refresh token response is invalid.")

    access = str(data.get("access_token") or "")
    refresh = str(data.get("refresh_token") or "")
    expires_in = int(data.get("expires_in") or 0)
    if not (access and refresh and expires_in):
        raise TokenRefreshError("Refresh token response missing required fields.")

    return TokenResponse(
        access=access,
        refresh=refresh,
        expires_at_ms=int(time.time() * 1000) + expires_in * 1000,
    )


class OAuthCallbackServer(HTTPServer):
    _oauth_result: dict[str, str] | None


class _CallbackHandler(BaseHTTPRequestHandler):
    server_version = "valuecell-openai-oauth"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]

        server = cast(OAuthCallbackServer, self.server)
        server._oauth_result = {"code": code, "state": state}

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h3>Login complete.</h3>You can close this tab.</body></html>"
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def run_local_callback_server(*, timeout_s: int = 180) -> dict[str, str] | None:
    try:
        server = OAuthCallbackServer(("127.0.0.1", 1455), _CallbackHandler)
    except OSError as exc:
        raise OAuthFlowError(
            "Port 1455 is unavailable. Close other Codex sessions or use manual login."
        ) from exc

    server._oauth_result = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    start = time.time()
    try:
        while time.time() - start < timeout_s:
            result = server._oauth_result
            if isinstance(result, dict) and result.get("code"):
                return {
                    "code": str(result.get("code")),
                    "state": str(result.get("state")),
                }
            time.sleep(0.1)
        return None
    finally:
        server.shutdown()
        server.server_close()


def parse_authorization_input(value: str) -> dict[str, str]:
    text = (value or "").strip()
    if not text:
        return {}

    try:
        url = urllib.parse.urlparse(text)
        if url.scheme and url.netloc:
            query = urllib.parse.parse_qs(url.query)
            return {
                "code": (query.get("code") or [""])[0],
                "state": (query.get("state") or [""])[0],
            }
    except Exception:
        pass

    if "code=" in text:
        query = urllib.parse.parse_qs(text)
        return {
            "code": (query.get("code") or [""])[0],
            "state": (query.get("state") or [""])[0],
        }

    if "#" in text:
        code, state = text.split("#", 1)
        return {"code": code, "state": state}

    return {"code": text}


def login_via_browser(*, timeout_s: int = 180) -> tuple[str, str] | None:
    verifier, challenge = generate_pkce()
    state = create_state()
    url = build_authorize_url(state=state, code_challenge=challenge)

    webbrowser.open(url)
    result = run_local_callback_server(timeout_s=timeout_s)
    if not result:
        return None

    if result.get("state") and result["state"] != state:
        raise OAuthFlowError("OAuth state mismatch.")

    return result["code"], verifier


def login_manual() -> tuple[str, str]:
    verifier, challenge = generate_pkce()
    state = create_state()
    url = build_authorize_url(state=state, code_challenge=challenge)

    print("Open this URL in your browser and complete login:\n")
    print(url)
    print("\nPaste the full redirect URL (or the authorization code) here:")
    parsed = parse_authorization_input(input("> "))
    code = parsed.get("code")
    if not code:
        raise OAuthFlowError("No authorization code provided.")

    if parsed.get("state") and parsed["state"] != state:
        raise OAuthFlowError("OAuth state mismatch.")

    return code, verifier


def login_and_store(
    *,
    auth_store: AuthStore | None = None,
    manual: bool = False,
    timeout_s: int = 180,
) -> OAuthCredentials:
    store = auth_store or AuthStore()

    result = login_manual() if manual else login_via_browser(timeout_s=timeout_s)
    if not result:
        raise OAuthFlowError("OAuth login timed out before completion.")

    code, verifier = result
    tokens = exchange_authorization_code(code=code, verifier=verifier)
    payload = decode_jwt_payload(tokens.access)
    if not payload:
        raise OAuthFlowError("OAuth access token is invalid.")

    account_id = extract_chatgpt_account_id(payload)
    if not account_id:
        raise OAuthFlowError("Failed to derive chatgpt account id from token.")

    creds = OAuthCredentials(
        access=tokens.access,
        refresh=tokens.refresh,
        expires=tokens.expires_at_ms,
        account_id=account_id,
    )
    store.save(creds)
    return creds


def logout(*, auth_store: AuthStore | None = None) -> None:
    (auth_store or AuthStore()).delete()


def has_stored_credentials(*, auth_store: AuthStore | None = None) -> bool:
    try:
        (auth_store or AuthStore()).load()
        return True
    except NotAuthenticatedError:
        return False


def get_oauth_status(*, auth_store: AuthStore | None = None) -> OAuthStatus:
    try:
        creds = (auth_store or AuthStore()).load()
    except NotAuthenticatedError:
        return OAuthStatus(authenticated=False)

    now_ms = int(time.time() * 1000)
    return OAuthStatus(
        authenticated=True,
        expires_at_ms=creds.expires,
        account_id=creds.account_id,
        expired=creds.expires <= now_ms,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="ValueCell ChatGPT OAuth helper")
    sub = parser.add_subparsers(dest="command", required=True)

    login_parser = sub.add_parser("login", help="Run interactive OAuth login")
    login_parser.add_argument("--manual", action="store_true")
    login_parser.add_argument("--timeout", type=int, default=180)
    sub.add_parser("status", help="Show OAuth status")
    sub.add_parser("logout", help="Delete stored OAuth credentials")

    args = parser.parse_args()
    try:
        if args.command == "login":
            creds = login_and_store(manual=args.manual, timeout_s=args.timeout)
            print(
                json.dumps(
                    {
                        "authenticated": True,
                        "account_id": creds.account_id,
                        "expires_at_ms": creds.expires,
                    },
                    indent=2,
                )
            )
        elif args.command == "status":
            print(json.dumps(get_oauth_status().__dict__, indent=2))
        elif args.command == "logout":
            logout()
            print(json.dumps({"authenticated": False}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
