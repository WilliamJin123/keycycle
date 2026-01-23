from .legacy_multi_provider_wrapper import MultiProviderWrapper
from .multi_client_wrapper import MultiClientWrapper, ProviderEnvConfig
from .key_rotation.rotation_manager import RotatingKeyManager
from .config.dataclasses import RateLimits, KeyLimitOverride
from .core.utils import KeyEntry
from .core.exceptions import (
    KeycycleError,
    NoAvailableKeyError,
    KeyNotFoundError,
    InvalidKeyError,
    RateLimitError,
    ConfigurationError,
)
from .adapters.generic_adapter import (
    create_rotating_client,
    detect_async_client,
    get_valid_constructor_kwargs,
    GenericClientConfig,
    SyncGenericRotatingClient,
    AsyncGenericRotatingClient,
)

__all__ = [
    # New primary wrapper (multi-provider support)
    "MultiClientWrapper",
    "ProviderEnvConfig",
    # Legacy wrapper (single provider, backward compatible)
    "MultiProviderWrapper",
    # Main classes
    "RateLimits",
    "KeyLimitOverride",
    "KeyEntry",
    "RotatingKeyManager",
    # Generic rotating client
    "create_rotating_client",
    "detect_async_client",
    "get_valid_constructor_kwargs",
    "GenericClientConfig",
    "SyncGenericRotatingClient",
    "AsyncGenericRotatingClient",
    # Exceptions
    "KeycycleError",
    "NoAvailableKeyError",
    "KeyNotFoundError",
    "InvalidKeyError",
    "RateLimitError",
    "ConfigurationError",
]