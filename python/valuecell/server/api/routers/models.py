"""Models API router: provide LLM model configuration defaults."""

import asyncio
import os
from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, HTTPException, Query

from valuecell.config.constants import CONFIG_DIR
from valuecell.config.loader import get_config_loader
from valuecell.config.manager import get_config_manager
from valuecell.openai_oauth.auth import get_oauth_status, login_and_store, logout
from valuecell.openai_oauth.model import OpenAIOAuthChat
from valuecell.utils.env import get_system_env_path

from ..schemas import SuccessResponse
from ..schemas.model import (
    AddModelRequest,
    CheckModelRequest,
    CheckModelResponse,
    ModelItem,
    ModelProviderSummary,
    ProviderDetailData,
    ProviderModelEntry,
    ProviderUpdateRequest,
    SetDefaultModelRequest,
    SetDefaultProviderRequest,
)

# Optional fallback constants from StrategyAgent
try:
    from valuecell.agents.common.trading.constants import (
        DEFAULT_AGENT_MODEL,
        DEFAULT_MODEL_PROVIDER,
    )
except Exception:  # pragma: no cover - constants may not exist in minimal env
    DEFAULT_MODEL_PROVIDER = "openai"
    DEFAULT_AGENT_MODEL = "gpt-5.3-codex"


def create_models_router() -> APIRouter:
    """Create models-related router with endpoints for model configs and provider management."""

    router = APIRouter(prefix="/models", tags=["Models"])

    # ---- Utility helpers (local to router) ----
    def _env_paths() -> List[Path]:
        """Return only system .env path for writes (single source of truth)."""
        system_env = get_system_env_path()
        return [system_env]

    def _set_env(key: str, value: str) -> bool:
        os.environ[key] = value
        updated_any = False
        for env_file in _env_paths():
            # Ensure parent directory exists for system env file
            try:
                env_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Best effort; continue even if directory creation fails
                pass
            lines: List[str] = []
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            updated = False
            found = False
            new_lines: List[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    found = True
                    updated = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"{key}={value}\n")
                updated = True
            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            updated_any = updated_any or updated
        return updated_any

    def _provider_yaml(provider: str) -> Path:
        return CONFIG_DIR / "providers" / f"{provider}.yaml"

    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _write_yaml(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def _refresh_configs() -> None:
        loader = get_config_loader()
        loader.clear_cache()
        manager = get_config_manager()
        manager._config = manager.loader.load_config()

    def _api_key_url_for(provider: str) -> str | None:
        """Return the URL for obtaining an API key for the given provider."""
        mapping = {
            "google": "https://aistudio.google.com/app/api-keys",
            "openrouter": "https://openrouter.ai/settings/keys",
            "openai": None,
            "azure": "https://azure.microsoft.com/en-us/products/ai-foundry/models/openai/",
            "siliconflow": "https://cloud.siliconflow.cn/account/ak",
            "deepseek": "https://platform.deepseek.com/api_keys",
            "dashscope": "https://bailian.console.aliyun.com/#/home",
            "ollama": None,
        }
        return mapping.get(provider)

    def _build_provider_detail(provider: str) -> ProviderDetailData:
        manager = get_config_manager()
        cfg = manager.get_provider_config(provider)
        if cfg is None:
            raise HTTPException(
                status_code=404, detail=f"Provider '{provider}' not found"
            )

        status = get_oauth_status() if provider == "openai" else None
        models_entries: List[ProviderModelEntry] = []
        for m in cfg.models or []:
            if isinstance(m, dict):
                mid = m.get("id")
                name = m.get("name")
                if mid:
                    models_entries.append(
                        ProviderModelEntry(model_id=mid, model_name=name)
                    )

        return ProviderDetailData(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            is_default=(cfg.name == manager.primary_provider),
            default_model_id=cfg.default_model,
            auth_type="oauth" if provider == "openai" else "api_key",
            oauth_authenticated=status.authenticated if status else False,
            oauth_expires_at=status.expires_at_ms if status else None,
            oauth_account_id=status.account_id if status else None,
            api_key_url=_api_key_url_for(cfg.name),
            models=models_entries,
        )

    @router.get(
        "/providers",
        response_model=SuccessResponse[List[ModelProviderSummary]],
        summary="List model providers",
        description="List available providers with status and basics.",
    )
    async def list_providers() -> SuccessResponse[List[ModelProviderSummary]]:
        try:
            items = [ModelProviderSummary(provider="openai")]
            return SuccessResponse.create(
                data=items, msg=f"Retrieved {len(items)} providers"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to list providers: {e}"
            )

    @router.get(
        "/providers/{provider}",
        response_model=SuccessResponse[ProviderDetailData],
        summary="Get provider details",
        description="Get configuration and models for a provider.",
    )
    async def get_provider_detail(provider: str) -> SuccessResponse[ProviderDetailData]:
        try:
            if provider != "openai":
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )
            detail = _build_provider_detail(provider)
            return SuccessResponse.create(
                data=detail, msg=f"Provider '{provider}' details"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get provider: {e}")

    @router.put(
        "/providers/{provider}/config",
        response_model=SuccessResponse[ProviderDetailData],
        summary="Update provider config",
        description="Update provider API key and host, then refresh configs.",
    )
    async def update_provider_config(
        provider: str, payload: ProviderUpdateRequest
    ) -> SuccessResponse[ProviderDetailData]:
        try:
            if provider != "openai":
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )
            loader = get_config_loader()
            provider_raw = loader.load_provider_config(provider)
            if not provider_raw:
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )

            connection = provider_raw.get("connection", {})
            api_key_env = connection.get("api_key_env")
            endpoint_env = connection.get("endpoint_env")

            if payload.api_key not in (None, ""):
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI uses ChatGPT OAuth in this build; API keys are disabled.",
                )

            # Update base_url via env when endpoint_env exists (Azure),
            # otherwise prefer updating the env placeholder if present; fallback to YAML
            # Accept empty string as a deliberate clear; skip only when field is omitted
            if payload.base_url is not None:
                if endpoint_env:
                    _set_env(endpoint_env, payload.base_url)
                else:
                    # Try to detect ${ENV_VAR:default} syntax in provider YAML
                    path = _provider_yaml(provider)
                    data = _load_yaml(path)
                    connection_raw = data.get("connection", {})
                    raw_base = connection_raw.get("base_url")
                    env_var_name = None
                    if (
                        isinstance(raw_base, str)
                        and raw_base.startswith("${")
                        and "}" in raw_base
                    ):
                        try:
                            inner = raw_base[2 : raw_base.index("}")]
                            env_var_name = inner.split(":", 1)[0]
                        except Exception:
                            env_var_name = None

                    if env_var_name:
                        _set_env(env_var_name, payload.base_url)
                    else:
                        data.setdefault("connection", {})
                        data["connection"]["base_url"] = payload.base_url
                        _write_yaml(path, data)

            _refresh_configs()

            detail = _build_provider_detail(provider)
            return SuccessResponse.create(
                data=detail, msg=f"Provider '{provider}' config updated"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to update provider config: {e}"
            )

    @router.post(
        "/providers/{provider}/models",
        response_model=SuccessResponse[ModelItem],
        summary="Add provider model",
        description="Add a model id to provider YAML.",
    )
    async def add_provider_model(
        provider: str, payload: AddModelRequest
    ) -> SuccessResponse[ModelItem]:
        try:
            path = _provider_yaml(provider)
            data = _load_yaml(path)
            if not data:
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )
            models = data.get("models") or []
            for m in models:
                if isinstance(m, dict) and m.get("id") == payload.model_id:
                    if payload.model_name:
                        m["name"] = payload.model_name
                    # If provider has no default model, set this one as default
                    existing_default = str(data.get("default_model", "")).strip()
                    if not existing_default:
                        data["default_model"] = payload.model_id
                    _write_yaml(path, data)
                    _refresh_configs()
                    return SuccessResponse.create(
                        data=ModelItem(
                            model_id=payload.model_id, model_name=m.get("name")
                        ),
                        msg=(
                            "Model already exists; updated model_name if provided"
                            + ("; set as default model" if not existing_default else "")
                        ),
                    )
            models.append(
                {"id": payload.model_id, "name": payload.model_name or payload.model_id}
            )
            data["models"] = models
            # If provider has no default model, set the added one as default
            existing_default = str(data.get("default_model", "")).strip()
            if not existing_default:
                data["default_model"] = payload.model_id
            _write_yaml(path, data)
            _refresh_configs()
            return SuccessResponse.create(
                data=ModelItem(
                    model_id=payload.model_id,
                    model_name=payload.model_name or payload.model_id,
                ),
                msg="Model added"
                + ("; set as default model" if not existing_default else ""),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add model: {e}")

    @router.delete(
        "/providers/{provider}/models",
        response_model=SuccessResponse[dict],
        summary="Remove provider model",
        description="Remove a model id from provider YAML.",
    )
    async def remove_provider_model(
        provider: str,
        model_id: str = Query(..., description="Model identifier to remove"),
    ) -> SuccessResponse[dict]:
        try:
            path = _provider_yaml(provider)
            data = _load_yaml(path)
            if not data:
                raise HTTPException(
                    status_code=500, detail=f"Provider '{provider}' not found"
                )
            models = data.get("models") or []
            before = len(models)
            models = [
                m
                for m in models
                if not (isinstance(m, dict) and m.get("id") == model_id)
            ]
            after = len(models)
            data["models"] = models
            _write_yaml(path, data)
            _refresh_configs()
            removed = before != after
            return SuccessResponse.create(
                data={"removed": removed, "remaining": after},
                msg="Model removed" if removed else "Model not found",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove model: {e}")

    @router.put(
        "/providers/default",
        response_model=SuccessResponse[dict],
        summary="Set default provider",
        description="Set PRIMARY_PROVIDER via env and refresh configs.",
    )
    async def set_default_provider(
        payload: SetDefaultProviderRequest,
    ) -> SuccessResponse[dict]:
        try:
            if payload.provider != "openai":
                raise HTTPException(
                    status_code=400,
                    detail="Only the OpenAI OAuth provider is supported in this build.",
                )
            _set_env("PRIMARY_PROVIDER", payload.provider)
            _refresh_configs()
            manager = get_config_manager()
            return SuccessResponse.create(
                data={"primary_provider": manager.primary_provider},
                msg=f"Primary provider set to '{payload.provider}'",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to set default provider: {e}"
            )

    @router.put(
        "/providers/{provider}/default-model",
        response_model=SuccessResponse[ProviderDetailData],
        summary="Set provider default model",
        description="Update provider default_model in YAML and refresh configs.",
    )
    async def set_provider_default_model(
        provider: str, payload: SetDefaultModelRequest
    ) -> SuccessResponse[ProviderDetailData]:
        try:
            path = _provider_yaml(provider)
            data = _load_yaml(path)
            if not data:
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )

            # Ensure the model exists in the list and optionally update name
            models = data.get("models") or []
            found = False
            for m in models:
                if isinstance(m, dict) and m.get("id") == payload.model_id:
                    if payload.model_name:
                        m["name"] = payload.model_name
                    found = True
                    break
            if not found:
                models.append(
                    {
                        "id": payload.model_id,
                        "name": payload.model_name or payload.model_id,
                    }
                )
            data["models"] = models

            # Set default model
            data["default_model"] = payload.model_id
            _write_yaml(path, data)
            _refresh_configs()

            # Build response from refreshed config
            detail = _build_provider_detail(provider)
            return SuccessResponse.create(
                data=detail,
                msg=(f"Default model for '{provider}' set to '{payload.model_id}'"),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to set default model: {e}"
            )

    @router.post(
        "/check",
        response_model=SuccessResponse[CheckModelResponse],
        summary="Check model availability",
        description=(
            "Perform a minimal live request to verify the model responds. "
            "This endpoint does not validate provider configuration or API key presence."
        ),
    )
    async def check_model(
        payload: CheckModelRequest,
    ) -> SuccessResponse[CheckModelResponse]:
        try:
            manager = get_config_manager()
            provider = payload.provider or manager.primary_provider
            if provider != "openai":
                raise HTTPException(
                    status_code=400,
                    detail="Only the OpenAI OAuth provider is supported in this build.",
                )
            cfg = manager.get_provider_config(provider)
            if cfg is None:
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )

            model_id = payload.model_id or cfg.default_model
            if not model_id:
                raise HTTPException(
                    status_code=400,
                    detail="Model id not specified and provider has no default",
                )

            # Perform a minimal live request (ping) without configuration validation
            result = CheckModelResponse(
                ok=False,
                provider=provider,
                model_id=model_id,
                status=None,
                error=None,
            )
            try:
                if not get_oauth_status().authenticated:
                    result.status = "auth_failed"
                    result.error = "ChatGPT OAuth is not connected"
                    return SuccessResponse.create(data=result, msg="Auth failed")
                model = OpenAIOAuthChat(
                    id=model_id,
                    base_url=cfg.base_url or "",
                )
                await asyncio.to_thread(model.probe)
                result.ok = True
                result.status = "reachable"
                return SuccessResponse.create(data=result, msg="Model reachable")
            except Exception as e:
                result.ok = False
                result.status = "request_failed"
                result.error = str(e)
                return SuccessResponse.create(data=result, msg="Request failed")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to check model: {e}")

    @router.post(
        "/providers/openai/oauth/login",
        response_model=SuccessResponse[ProviderDetailData],
        summary="Run OpenAI OAuth login",
        description="Open the ChatGPT OAuth flow in a browser and store local credentials.",
    )
    async def login_openai_oauth(
        manual: bool = Query(
            False,
            description="Use manual login flow instead of the local callback server",
        ),
        timeout_s: int = Query(180, description="OAuth login timeout in seconds"),
    ) -> SuccessResponse[ProviderDetailData]:
        try:
            await asyncio.to_thread(
                login_and_store,
                manual=manual,
                timeout_s=timeout_s,
            )
            _refresh_configs()
            return SuccessResponse.create(
                data=_build_provider_detail("openai"),
                msg="OpenAI OAuth login completed",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OAuth login failed: {e}")

    @router.post(
        "/providers/openai/oauth/logout",
        response_model=SuccessResponse[ProviderDetailData],
        summary="Delete OpenAI OAuth credentials",
        description="Remove locally stored ChatGPT OAuth credentials.",
    )
    async def logout_openai_oauth() -> SuccessResponse[ProviderDetailData]:
        try:
            await asyncio.to_thread(logout)
            _refresh_configs()
            return SuccessResponse.create(
                data=_build_provider_detail("openai"),
                msg="OpenAI OAuth credentials removed",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OAuth logout failed: {e}")

    return router
