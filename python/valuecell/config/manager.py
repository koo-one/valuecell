"""
Configuration Manager - High-level interface for accessing configurations

Provides a clean API for:
- Getting provider configurations with API key validation
- Getting agent configurations with full three-tier override
- Validating configurations
- Listing available providers and models
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from valuecell.config.loader import ConfigLoader, get_config_loader
from valuecell.openai_oauth.auth import has_stored_credentials

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Provider configuration with resolved values"""

    name: str
    enabled: bool
    api_key: Optional[str]
    base_url: Optional[str]
    default_model: str
    models: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    # Embedding support
    default_embedding_model: Optional[str] = None
    embedding_models: List[Dict[str, Any]] = None
    embedding_parameters: Dict[str, Any] = None
    extra_config: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values for optional fields"""
        if self.embedding_models is None:
            self.embedding_models = []
        if self.embedding_parameters is None:
            self.embedding_parameters = {}
        if self.extra_config is None:
            self.extra_config = {}


@dataclass
class AgentModelConfig:
    """Agent model configuration with provider-specific model mappings"""

    model_id: str
    provider: str
    parameters: Dict[str, Any]
    # Provider-specific model mappings for fallback
    # e.g., {"siliconflow": "deepseek-ai/DeepSeek-V3.1-Terminus", "google": "gemini-2.0-flash"}
    provider_models: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Initialize default values for optional fields"""
        if self.provider_models is None:
            self.provider_models = {}


@dataclass
class AgentConfig:
    """Complete agent configuration"""

    name: str
    enabled: bool
    primary_model: AgentModelConfig
    embedding_model: Optional[AgentModelConfig]
    api_keys: Dict[str, Dict[str, Any]]
    capabilities: Dict[str, Any]
    extra_config: Dict[str, Any]


class ConfigManager:
    """
    High-level configuration manager

    Provides easy access to configurations with:
    - API key validation
    - Provider capability checking
    - Agent configuration resolution
    """

    def __init__(self, loader: Optional[ConfigLoader] = None):
        """
        Initialize configuration manager

        Args:
            loader: ConfigLoader instance (auto-created if None)
        """
        self.loader = loader or get_config_loader()
        self._config = self.loader.load_config()

    @property
    def app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return self._config.get("app", {})

    @property
    def primary_provider(self) -> str:
        """
        Get primary model provider.

        ValueCell is intentionally pinned to the local OpenAI OAuth path.
        """
        env_provider = os.getenv("PRIMARY_PROVIDER")
        if env_provider:
            if env_provider != "openai":
                logger.warning(
                    "Ignoring PRIMARY_PROVIDER=%s because ValueCell is configured for OpenAI OAuth only.",
                    env_provider,
                )
            else:
                logger.debug("Using provider from PRIMARY_PROVIDER: openai")
        return "openai"

    @property
    def fallback_providers(self) -> List[str]:
        """Return no fallback providers.

        The application is intentionally restricted to OpenAI OAuth.
        """
        return []

    def get_provider_config(
        self, provider_name: Optional[str] = None
    ) -> Optional[ProviderConfig]:
        """
        Get provider configuration with API key validation

        Args:
            provider_name: Provider name (uses primary if None)

        Returns:
            ProviderConfig or None if not found/not configured
        """
        provider_name = provider_name or self.primary_provider

        # Load provider YAML
        provider_data = self.loader.load_provider_config(provider_name)

        if not provider_data:
            logger.warning(f"Provider config not found: {provider_name}")
            return None

        # Get connection info
        connection = provider_data.get("connection", {})

        # Get API key from environment
        api_key_env = connection.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None

        # Get endpoint for Azure
        endpoint_env = connection.get("endpoint_env")
        base_url = connection.get("base_url")
        if endpoint_env:
            base_url = os.getenv(endpoint_env) or base_url

        # Get default model
        default_model = provider_data.get("default_model", "")

        # Get model list
        models = provider_data.get("models", [])

        # Get default parameters
        defaults = provider_data.get("defaults", {})

        # Check if enabled
        enabled = provider_data.get("enabled", True)

        # Get embedding configuration
        embedding_config = provider_data.get("embedding", {})
        default_embedding_model = embedding_config.get("default_model")
        embedding_models = embedding_config.get("models", [])
        embedding_defaults = embedding_config.get("defaults", {})

        return ProviderConfig(
            name=provider_name,
            enabled=enabled,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            models=models,
            parameters=defaults,
            default_embedding_model=default_embedding_model,
            embedding_models=embedding_models,
            embedding_parameters=embedding_defaults,
            extra_config={
                k: v
                for k, v in provider_data.items()
                if k
                not in [
                    "connection",
                    "default_model",
                    "models",
                    "defaults",
                    "enabled",
                    "embedding",
                ]
            },
        )

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get complete agent configuration with all overrides

        This applies the full three-tier override system:
        1. Agent YAML (developer defaults)
        2. .env file (user preferences)
        3. Environment variables (runtime overrides)

        Args:
            agent_name: Agent name

        Returns:
            AgentConfig or None if not found
        """
        # Load agent config with all overrides applied
        agent_data = self.loader.load_agent_config(agent_name)

        if not agent_data:
            logger.warning(f"Agent config not found: {agent_name}")
            return None

        # Extract model configuration
        models = agent_data.get("models", {})
        primary = models.get("primary", {})

        # Get model ID (with fallback chain)
        model_id = primary.get("model_id")
        provider = primary.get("provider") or self.primary_provider

        # If model_id is None, use provider's default
        if not model_id:
            provider_config = self.get_provider_config(provider)
            if provider_config:
                model_id = provider_config.default_model

        # Get parameters
        parameters = primary.get("parameters") or {}

        # Merge with global defaults
        global_models = self._config.get("models") or {}
        global_defaults = global_models.get("defaults") or {}
        merged_params = {**global_defaults, **parameters}

        # Get provider-specific model mappings
        provider_models = primary.get("provider_models", {})

        primary_model = AgentModelConfig(
            model_id=model_id or "",
            provider=provider,
            parameters=merged_params,
            provider_models=provider_models,
        )

        # Extract embedding model config if present
        embedding_model = None
        embedding_data = models.get("embedding")
        if embedding_data:
            embedding_provider_models = embedding_data.get("provider_models", {})
            embedding_model = AgentModelConfig(
                model_id=embedding_data.get("model_id", ""),
                provider=embedding_data.get("provider", "openai"),
                parameters=embedding_data.get("parameters", {}),
                provider_models=embedding_provider_models,
            )

        # Extract API keys
        api_keys = agent_data.get("api_keys", {})

        # Extract capabilities
        capabilities = agent_data.get("capabilities", {})

        return AgentConfig(
            name=agent_name,
            enabled=agent_data.get("enabled", True),
            primary_model=primary_model,
            embedding_model=embedding_model,
            api_keys=api_keys,
            capabilities=capabilities,
            extra_config={
                k: v
                for k, v in agent_data.items()
                if k not in ["name", "enabled", "models", "api_keys", "capabilities"]
            },
        )

    def get_enabled_providers(self) -> List[str]:
        """
        Get list of enabled providers with valid credentials.

        ValueCell currently exposes only the OpenAI OAuth provider.
        """
        provider_config = self.get_provider_config("openai")
        if provider_config and provider_config.enabled and has_stored_credentials():
            return ["openai"]
        return []

    def validate_provider(self, provider_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate provider configuration

        Args:
            provider_name: Provider to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if provider_name != "openai":
            return (
                False,
                f"Provider '{provider_name}' is disabled in this build. Use 'openai' with ChatGPT OAuth.",
            )

        provider_config = self.get_provider_config(provider_name)

        if not provider_config:
            return False, f"Provider '{provider_name}' not found in configuration"

        if not provider_config.enabled:
            return False, f"Provider '{provider_name}' is disabled in config"

        if provider_name == "openai":
            if not has_stored_credentials():
                return (
                    False,
                    "OpenAI OAuth is not configured. Sign in with ChatGPT first.",
                )
            return True, None

        # Check API key (except for ollama)
        if provider_name != "ollama" and not provider_config.api_key:
            # Get the env var name for helpful error message
            provider_data = self.loader.load_provider_config(provider_name)
            api_key_env = provider_data.get("connection", {}).get("api_key_env")
            return (
                False,
                f"API key not found for '{provider_name}'. Please set {api_key_env} in .env",
            )

        # Azure needs endpoint too
        if provider_name == "azure" and not provider_config.base_url:
            return (
                False,
                "Azure endpoint not configured. Please set AZURE_OPENAI_ENDPOINT",
            )

        return True, None

    def validate_agent(self, agent_name: str) -> tuple[bool, List[str]]:
        """
        Validate complete agent configuration

        Args:
            agent_name: Agent to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.loader.validate_agent_config(agent_name)

    def get_available_models(
        self, provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available models for a provider

        Args:
            provider: Provider name (uses primary if None)

        Returns:
            List of model dictionaries with metadata
        """
        provider_config = self.get_provider_config(provider)

        if not provider_config:
            return []

        return provider_config.models


# ============================================
# Singleton Instance
# ============================================

_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get singleton configuration manager

    Returns:
        ConfigManager instance
    """
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager
