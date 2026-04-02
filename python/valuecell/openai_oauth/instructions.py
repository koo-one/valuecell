"""Fetch and cache official Codex instruction prompts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

from valuecell.openai_oauth.store import get_home_dir

GITHUB_API_RELEASES = "https://api.github.com/repos/openai/codex/releases/latest"
GITHUB_HTML_RELEASES = "https://github.com/openai/codex/releases/latest"
RAW_PROMPT_URL = (
    "https://raw.githubusercontent.com/openai/codex/{ref}/codex-rs/core/{prompt_file}"
)
MODELS_URL = "https://raw.githubusercontent.com/openai/codex/{ref}/codex-rs/core/models.json"
INSTRUCTIONS_MODE_ENV = "VALUECELL_OPENAI_OAUTH_INSTRUCTIONS_MODE"

InstructionsMode = Literal["auto", "cache", "github"]


@dataclass(frozen=True)
class PromptFamily:
    family: str
    prompt_file: str | None
    cache_file: str


PROMPT_FAMILIES: list[PromptFamily] = [
    PromptFamily(
        family="gpt-5.3-codex",
        prompt_file=None,
        cache_file="gpt-5.3-codex-instructions.md",
    ),
    PromptFamily(
        family="codex",
        prompt_file="gpt_5_codex_prompt.md",
        cache_file="codex-instructions.md",
    ),
    PromptFamily(
        family="gpt-5.4",
        prompt_file=None,
        cache_file="gpt-5.4-instructions.md",
    ),
    PromptFamily(
        family="gpt-5.1",
        prompt_file="gpt_5_1_prompt.md",
        cache_file="gpt-5.1-instructions.md",
    ),
]


def normalize_model(model: str) -> str:
    model_id = model.split("/", 1)[1] if "/" in model else model
    return model_id.strip()


def get_model_family(model: str) -> PromptFamily:
    model_id = normalize_model(model).lower()
    if "gpt-5.3-codex" in model_id or "gpt 5.3 codex" in model_id:
        return PROMPT_FAMILIES[0]
    if "codex" in model_id:
        return PROMPT_FAMILIES[1]
    if "gpt-5.4" in model_id or "gpt 5.4" in model_id:
        return PROMPT_FAMILIES[2]
    return PROMPT_FAMILIES[3]


def get_instructions_mode() -> InstructionsMode:
    raw = (os.environ.get(INSTRUCTIONS_MODE_ENV) or "auto").strip().lower()
    if raw in {"auto", "cache", "github"}:
        return raw  # type: ignore[return-value]
    return "auto"


def get_cache_dir() -> Path:
    return get_home_dir() / "cache"


def _cache_paths(family: PromptFamily) -> tuple[Path, Path]:
    cache_dir = get_cache_dir()
    cache_path = cache_dir / family.cache_file
    meta_path = cache_dir / family.cache_file.replace(".md", "-meta.json")
    return cache_path, meta_path


def _latest_release_tag(session: requests.Session) -> str | None:
    try:
        response = session.get(GITHUB_API_RELEASES, timeout=15.0)
        if response.ok:
            data = response.json()
            tag = data.get("tag_name") if isinstance(data, dict) else None
            if isinstance(tag, str) and tag:
                return tag
    except Exception:
        pass

    try:
        response = session.get(
            GITHUB_HTML_RELEASES,
            allow_redirects=True,
            timeout=15.0,
        )
        response.raise_for_status()
        final_url = str(response.url)
        if "/tag/" in final_url:
            tag = final_url.rsplit("/tag/", 1)[-1]
            if tag and "/" not in tag:
                return tag

        marker = "/openai/codex/releases/tag/"
        text = response.text
        idx = text.find(marker)
        if idx >= 0:
            tail = text[idx + len(marker) :]
            tag = tail.split('"', 1)[0]
            if tag:
                return tag
    except Exception:
        return None

    return None


def _write_cache(cache_path: Path, meta_path: Path, *, text: str, source_url: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"source_url": source_url}, indent=2) + "\n",
        encoding="utf-8",
    )


def _fetch_base_instructions_from_models_json(
    session: requests.Session,
    *,
    ref: str,
    slug: str,
) -> tuple[str, str] | None:
    url = MODELS_URL.format(ref=ref)
    response = session.get(url, timeout=30.0)
    response.raise_for_status()

    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None

    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("slug") != slug:
            continue
        instructions = model.get("base_instructions")
        if isinstance(instructions, str) and instructions:
            return instructions, url

    return None


def get_codex_instructions(
    *,
    model: str,
    session: requests.Session | None = None,
) -> str:
    """Return the official Codex instruction prompt for a model family."""
    family = get_model_family(model)
    cache_path, meta_path = _cache_paths(family)
    mode = get_instructions_mode()

    if mode in {"auto", "cache"} and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    if mode == "cache":
        raise RuntimeError(
            f"Instructions cache is missing ({cache_path}). "
            f"Delete `{INSTRUCTIONS_MODE_ENV}=cache` or populate the cache first."
        )

    http = session or requests.Session()
    close_after = session is None
    try:
        release_tag = _latest_release_tag(http)
        last_error: Exception | None = None

        if family.prompt_file:
            candidate_urls: list[str] = []
            if release_tag:
                candidate_urls.append(
                    RAW_PROMPT_URL.format(ref=release_tag, prompt_file=family.prompt_file)
                )
            candidate_urls.append(
                RAW_PROMPT_URL.format(ref="main", prompt_file=family.prompt_file)
            )

            for url in candidate_urls:
                try:
                    response = http.get(url, timeout=30.0)
                    response.raise_for_status()
                    _write_cache(
                        cache_path,
                        meta_path,
                        text=response.text,
                        source_url=url,
                    )
                    return response.text
                except Exception as exc:
                    last_error = exc

        for ref in [candidate for candidate in (release_tag, "main") if candidate]:
            try:
                result = _fetch_base_instructions_from_models_json(
                    http,
                    ref=ref,
                    slug=family.family,
                )
                if result is None:
                    continue
                text, source_url = result
                _write_cache(cache_path, meta_path, text=text, source_url=source_url)
                return text
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            "Failed to fetch Codex instructions from official openai/codex releases. "
            f"Checked cache path: {cache_path}"
        ) from last_error
    finally:
        if close_after:
            http.close()
