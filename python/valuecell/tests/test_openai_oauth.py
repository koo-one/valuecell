from __future__ import annotations

import pytest

from valuecell.config import manager as manager_mod
from valuecell.openai_oauth.exceptions import NotAuthenticatedError
from valuecell.openai_oauth.model import _iter_sse_events, _parse_assistant_response
from valuecell.openai_oauth.store import AuthStore, OAuthCredentials


def test_auth_store_roundtrip(tmp_path):
    auth_path = tmp_path / "auth.json"
    store = AuthStore(auth_path)
    creds = OAuthCredentials(
        access="access-token",
        refresh="refresh-token",
        expires=1234567890,
        account_id="account_123",
    )

    store.save(creds)

    assert store.load() == creds

    store.delete()
    with pytest.raises(NotAuthenticatedError):
        store.load()


def test_iter_sse_events_accepts_bytes():
    events = list(
        _iter_sse_events(
            [
                b'data: {"type":"response.completed","response":{"id":"resp_1","output_text":"hello"}}\n',
                b"\n",
            ]
        )
    )

    assert len(events) == 1
    assert events[0]["type"] == "response.completed"
    assert events[0]["response"]["output_text"] == "hello"


def test_parse_assistant_response_extracts_text_and_tool_calls():
    parsed = _parse_assistant_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Summary"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_quote",
                    "arguments": '{"ticker":"NVDA"}',
                },
            ]
        }
    )

    assert parsed["content"] == "Summary"
    assert parsed["tool_calls"][0]["id"] == "call_1"
    assert parsed["tool_calls"][0]["function"]["name"] == "lookup_quote"


def test_config_manager_openai_oauth_only(monkeypatch, tmp_path):
    auth_path = tmp_path / "openai.json"
    monkeypatch.setenv("VALUECELL_OPENAI_OAUTH_AUTH_PATH", str(auth_path))
    manager_mod._manager = None

    manager = manager_mod.get_config_manager()
    ok, error = manager.validate_provider("openai")
    assert not ok
    assert "OAuth" in (error or "")

    AuthStore(auth_path).save(
        OAuthCredentials(
            access="access-token",
            refresh="refresh-token",
            expires=9999999999999,
            account_id="account_123",
        )
    )
    manager_mod._manager = None

    manager = manager_mod.get_config_manager()
    ok, error = manager.validate_provider("openai")
    assert ok
    assert error is None

    ok, error = manager.validate_provider("google")
    assert not ok
    assert "disabled" in (error or "")
