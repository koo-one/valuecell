"""Credential storage for local ChatGPT OAuth support."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from valuecell.openai_oauth.exceptions import NotAuthenticatedError
from valuecell.utils.env import get_system_env_dir

HOME_ENV = "VALUECELL_OPENAI_OAUTH_HOME"
AUTH_PATH_ENV = "VALUECELL_OPENAI_OAUTH_AUTH_PATH"


def get_home_dir() -> Path:
    """Return the configured home directory for OAuth state."""
    env_home = os.environ.get(HOME_ENV)
    if env_home:
        return Path(env_home).expanduser()
    return get_system_env_dir() / "openai_oauth"


def get_default_auth_path() -> Path:
    """Return the preferred auth file path."""
    env_path = os.environ.get(AUTH_PATH_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return get_home_dir() / "auth" / "openai.json"


@dataclass(frozen=True)
class OAuthCredentials:
    access: str
    refresh: str
    expires: int
    account_id: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OAuthCredentials":
        access = str(data.get("access") or "")
        refresh = str(data.get("refresh") or "")
        account_id = str(data.get("account_id") or "")

        expires_raw = data.get("expires")
        if isinstance(expires_raw, (int, float)):
            expires = int(expires_raw)
        elif isinstance(expires_raw, str):
            try:
                expires = int(expires_raw)
            except ValueError:
                expires = 0
        else:
            expires = 0

        return cls(
            access=access,
            refresh=refresh,
            expires=expires,
            account_id=account_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "oauth",
            "access": self.access,
            "refresh": self.refresh,
            "expires": self.expires,
            "account_id": self.account_id,
        }


class AuthStore:
    """Load and persist OAuth credentials."""

    def __init__(self, auth_path: Path | None = None) -> None:
        self.auth_path = auth_path or get_default_auth_path()

    def load(self) -> OAuthCredentials:
        if not self.auth_path.exists():
            raise NotAuthenticatedError("Not authenticated. Run the OAuth login flow.")

        try:
            raw = self.auth_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            raise NotAuthenticatedError(
                "Auth file is invalid. Run the OAuth login flow again."
            ) from exc
        if not isinstance(data, dict):
            raise NotAuthenticatedError("Auth file is invalid. Run the OAuth login flow again.")

        creds = OAuthCredentials.from_dict(data)
        if not (creds.access and creds.refresh and creds.account_id and creds.expires):
            raise NotAuthenticatedError("Auth file is incomplete. Run the OAuth login flow again.")
        return creds

    def save(self, creds: OAuthCredentials) -> None:
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.auth_path.with_suffix(self.auth_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(creds.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.auth_path)
        self._chmod_user_only(self.auth_path)

    def delete(self) -> None:
        if self.auth_path.exists():
            self.auth_path.unlink()

    @staticmethod
    def _chmod_user_only(path: Path) -> None:
        if os.name == "nt":
            return

        try:
            current = path.stat().st_mode
            path.chmod(
                (current & ~stat.S_IRWXG & ~stat.S_IRWXO)
                | stat.S_IRUSR
                | stat.S_IWUSR
            )
        except OSError:
            return
