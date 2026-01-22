"""
Multi-client wrapper for managing multiple providers with rotating API keys.

This module provides the primary interface for keycycle, supporting:
- Multi-parameter key rotation (api_key + index_id, etc.)
- Multiple providers in a single wrapper
- Environment-based configuration with indexed params
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from dotenv import load_dotenv

from .key_rotation.rotation_manager import RotatingKeyManager
from .config.dataclasses import RateLimits, KeyLimitOverride
from .config.enums import RateLimitStrategy
from .config.models import MODEL_LIMITS, PROVIDER_STRATEGIES
from .core.utils import (
    KeyEntry,
    get_key_suffix,
    load_api_keys as _load_api_keys,
    normalize_key_limits as _normalize_key_limits,
)
from .usage.db_logic import UsageDatabase
from .adapters.generic_adapter import (
    create_rotating_client,
    SyncGenericRotatingClient,
    AsyncGenericRotatingClient,
)

T = TypeVar("T")


@dataclass
class ProviderConfig:
    """Internal config for a registered provider."""
    default_model: Optional[str] = None
    limits: Optional[Dict[str, RateLimits]] = None
    excluded_kwargs: Optional[List[str]] = None


@dataclass
class ProviderEnvConfig:
    """
    Configuration for loading a provider from environment variables.

    Attributes:
        default_model: Default model ID for this provider
        extra_params: List of extra parameter names to load alongside API keys.
            For each param, loads {PROVIDER}_{PARAM}_N environment variables.
        strategy: Rate limit strategy (per-model or global)
        limits: Custom rate limits per model
        api_key_param: Name of the API key parameter (default: "api_key")
        excluded_kwargs: List of kwarg names to exclude from client constructor.
            Useful for clients that don't accept certain params (e.g., TwelveLabs doesn't accept 'model').
    """
    default_model: Optional[str] = None
    extra_params: Optional[List[str]] = None
    strategy: RateLimitStrategy = RateLimitStrategy.PER_MODEL
    limits: Optional[Dict[str, RateLimits]] = None
    api_key_param: str = "api_key"
    excluded_kwargs: Optional[List[str]] = None


class MultiClientWrapper:
    """
    Manages multiple providers with rotating API keys.

    This is the primary wrapper for keycycle, supporting:
    - Multiple providers in a single instance
    - Multi-parameter rotation (api_key + other params)
    - Environment-based configuration

    Example:
        >>> wrapper = MultiClientWrapper.from_env({
        ...     "twelvelabs": ProviderEnvConfig(
        ...         default_model="pegasus-1",
        ...         extra_params=["index_id"],
        ...     ),
        ...     "anthropic": ProviderEnvConfig(
        ...         default_model="claude-3-sonnet",
        ...     ),
        ... })
        >>> tl_client = wrapper.get_rotating_client("twelvelabs", TwelveLabs)
        >>> claude_client = wrapper.get_rotating_client("anthropic", Anthropic)
    """

    MODEL_LIMITS = MODEL_LIMITS
    PROVIDER_STRATEGIES = PROVIDER_STRATEGIES

    def __init__(self, db_url: Optional[str] = None, db_env_var: str = "TIDB_DB_URL"):
        """
        Initialize a MultiClientWrapper.

        Args:
            db_url: Database URL for usage persistence (optional)
            db_env_var: Environment variable name for database URL
        """
        self.db = UsageDatabase(db_url, db_env_var)
        self._managers: Dict[str, RotatingKeyManager] = {}
        self._configs: Dict[str, ProviderConfig] = {}
        self._key_limits: Dict[str, Dict[str, KeyLimitOverride]] = {}

    def register_provider(
        self,
        provider: str,
        keys: List[KeyEntry],
        strategy: Optional[RateLimitStrategy] = None,
        default_model: Optional[str] = None,
        limits: Optional[Dict[str, RateLimits]] = None,
        api_key_param: str = "api_key",
        key_limits: Optional[Dict[Union[int, str], KeyLimitOverride]] = None,
        **kwargs
    ) -> "MultiClientWrapper":
        """
        Register a provider with its keys.

        Args:
            provider: Provider name (e.g., "twelvelabs", "anthropic")
            keys: List of key entries (strings or dicts with params)
            strategy: Rate limit strategy (defaults to provider's default)
            default_model: Default model ID
            limits: Custom rate limits per model
            api_key_param: Name of the API key parameter
            key_limits: Per-key rate limit overrides
            **kwargs: Additional arguments for RotatingKeyManager

        Returns:
            Self for method chaining
        """
        provider = provider.lower()

        # Use provider-specific strategy if not specified
        if strategy is None:
            strategy = self.PROVIDER_STRATEGIES.get(provider, RateLimitStrategy.PER_MODEL)

        # Normalize key_limits to suffix-based
        normalized_key_limits = self._normalize_key_limits_from_entries(keys, key_limits, api_key_param)
        self._key_limits[provider] = normalized_key_limits

        manager = RotatingKeyManager(
            api_keys=keys,
            provider_name=provider,
            strategy=strategy,
            db=self.db,
            api_key_param=api_key_param,
            limit_resolver=lambda m, k, p=provider: self._resolve_limits(p, m, k),
            **kwargs
        )
        self._managers[provider] = manager
        self._configs[provider] = ProviderConfig(default_model, limits)
        return self

    def _normalize_key_limits_from_entries(
        self,
        keys: List[KeyEntry],
        key_limits: Optional[Dict[Union[int, str], KeyLimitOverride]],
        api_key_param: str = "api_key",
    ) -> Dict[str, KeyLimitOverride]:
        """Normalize key_limits to suffix-based lookup."""
        if not key_limits:
            return {}

        # Extract primary keys
        primary_keys = []
        for entry in keys:
            if isinstance(entry, str):
                primary_keys.append(entry)
            else:
                primary_keys.append(entry.get(api_key_param, ""))

        return _normalize_key_limits(primary_keys, key_limits)

    def _resolve_limits(
        self,
        provider: str,
        model_id: str,
        key_suffix: Optional[str] = None
    ) -> RateLimits:
        """Resolve rate limits for a provider/model/key combination."""
        config = self._configs.get(provider)
        provider_key_limits = self._key_limits.get(provider, {})

        # Check key-specific overrides first
        if key_suffix and provider_key_limits:
            override = provider_key_limits.get(key_suffix)
            if override:
                if isinstance(override, RateLimits):
                    return override
                elif isinstance(override, dict):
                    if model_id in override:
                        return override[model_id]
                    if '__default__' in override:
                        return override['__default__']

        # Check provider-specific limits from config
        if config and config.limits and model_id in config.limits:
            return config.limits[model_id]

        # Fall back to global model limits
        provider_limits = self.MODEL_LIMITS.get(provider, {})
        return provider_limits.get(model_id, provider_limits.get('default', RateLimits(10, 100, 1000)))

    def get_rotating_client(
        self,
        provider: str,
        client_class: Type[T],
        model: Optional[str] = None,
        api_key_param: Optional[str] = None,
        is_async: Optional[bool] = None,
        usage_extractor: Optional[Callable[[Any], int]] = None,
        estimated_tokens: int = 1000,
        max_retries: int = 5,
        model_param: str = "model",
        excluded_kwargs: Optional[List[str]] = None,
        **client_kwargs
    ) -> Union[SyncGenericRotatingClient[T], AsyncGenericRotatingClient[T]]:
        """
        Get a rotating client for a registered provider.

        Args:
            provider: Registered provider name
            client_class: The client class to wrap
            model: Model ID (defaults to provider's default_model)
            api_key_param: Override the API key parameter name
            is_async: Whether the client is async (auto-detected if None)
            usage_extractor: Custom function to extract token usage
            estimated_tokens: Estimated tokens per request
            max_retries: Max key rotations on rate limit errors
            model_param: Name of the model parameter in API calls
            excluded_kwargs: List of kwarg names to exclude from client constructor.
                Useful for clients that don't accept certain params (e.g., 'model' for TwelveLabs).
            **client_kwargs: Additional kwargs for client constructor

        Returns:
            A rotating client wrapper

        Raises:
            ValueError: If provider is not registered
        """
        provider = provider.lower()
        manager = self._managers.get(provider)
        if not manager:
            raise ValueError(f"Provider '{provider}' not registered. Call register_provider() first.")

        config = self._configs[provider]
        default_model = model or config.default_model or "default"

        # Use the manager's api_key_param if not overridden
        if api_key_param is None:
            api_key_param = manager.api_key_param

        # Use config's excluded_kwargs as default if not specified
        if excluded_kwargs is None:
            excluded_kwargs = config.excluded_kwargs

        return create_rotating_client(
            client_class=client_class,
            manager=manager,
            limit_resolver=lambda m, k: self._resolve_limits(provider, m, k),
            default_model=default_model,
            api_key_param=api_key_param,
            is_async=is_async,
            usage_extractor=usage_extractor,
            estimated_tokens=estimated_tokens,
            max_retries=max_retries,
            model_param=model_param,
            excluded_kwargs=excluded_kwargs,
            **client_kwargs,
        )

    def get_manager(self, provider: str) -> RotatingKeyManager:
        """Get the RotatingKeyManager for a provider."""
        provider = provider.lower()
        manager = self._managers.get(provider)
        if not manager:
            raise ValueError(f"Provider '{provider}' not registered.")
        return manager

    @classmethod
    def from_env(
        cls,
        providers: Dict[str, ProviderEnvConfig],
        env_file: Optional[str] = None,
        db_url: Optional[str] = None,
        db_env_var: str = "TIDB_DB_URL",
    ) -> "MultiClientWrapper":
        """
        Create a wrapper from environment variables.

        Args:
            providers: Dict mapping provider names to their env configs
            env_file: Path to .env file (optional)
            db_url: Database URL for usage persistence
            db_env_var: Environment variable name for database URL

        Returns:
            Configured MultiClientWrapper instance

        Example:
            >>> wrapper = MultiClientWrapper.from_env({
            ...     "twelvelabs": ProviderEnvConfig(
            ...         default_model="pegasus-1",
            ...         extra_params=["index_id"],
            ...     ),
            ...     "anthropic": ProviderEnvConfig(
            ...         default_model="claude-3-sonnet",
            ...     ),
            ... })

        Environment variables format:
            NUM_TWELVELABS=2
            TWELVELABS_API_KEY_1=key1
            TWELVELABS_INDEX_ID_1=idx_abc
            TWELVELABS_API_KEY_2=key2
            TWELVELABS_INDEX_ID_2=idx_xyz
        """
        instance = cls(db_url=db_url, db_env_var=db_env_var)

        for provider, config in providers.items():
            keys = cls.load_api_keys(
                provider,
                env_file=env_file,
                extra_params=config.extra_params,
                api_key_param=config.api_key_param,
            )
            instance.register_provider(
                provider=provider,
                keys=keys,
                default_model=config.default_model,
                strategy=config.strategy,
                limits=config.limits,
                api_key_param=config.api_key_param,
            )
            # Store excluded_kwargs from env config
            if config.excluded_kwargs:
                instance._configs[provider.lower()].excluded_kwargs = config.excluded_kwargs
        return instance

    @staticmethod
    def load_api_keys(
        provider: str,
        env_file: Optional[str] = None,
        extra_params: Optional[List[str]] = None,
        api_key_param: str = "api_key",
    ) -> List[KeyEntry]:
        """
        Load keys from environment variables with indexed pattern.

        Args:
            provider: Provider name (e.g., "twelvelabs")
            env_file: Path to .env file (optional)
            extra_params: List of extra param names to load
            api_key_param: Name for the API key in returned dicts

        Returns:
            List of key entries

        Environment variables format:
            NUM_{PROVIDER}=N
            {PROVIDER}_API_KEY_1, {PROVIDER}_API_KEY_2, ...
            {PROVIDER}_{PARAM}_1, {PROVIDER}_{PARAM}_2, ... (for each extra_param)
        """
        return _load_api_keys(provider, env_file, extra_params, api_key_param)

    def stop(self) -> None:
        """Stop all managers and flush logs."""
        for manager in self._managers.values():
            manager.stop()
