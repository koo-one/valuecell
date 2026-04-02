"""Exceptions for local ChatGPT OAuth support."""


class OAuthFlowError(RuntimeError):
    """Raised when the interactive OAuth flow fails."""


class TokenRefreshError(RuntimeError):
    """Raised when a stored refresh token can no longer be refreshed."""


class NotAuthenticatedError(RuntimeError):
    """Raised when local OAuth credentials are missing or invalid."""


class CodexAPIError(RuntimeError):
    """Raised when the ChatGPT Codex backend returns an error."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
