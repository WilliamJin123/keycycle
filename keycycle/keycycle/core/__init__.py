"""Core utilities and exceptions for keycycle."""
from .exceptions import (
    KeycycleError,
    KeycycleKeyError,
    NoAvailableKeyError,
    KeyNotFoundError,
    InvalidKeyError,
    RateLimitError,
    ConfigurationError,
    MissingEnvironmentVariableError,
    InvalidConfigurationError,
)
from .utils import (
    get_key_suffix,
    is_rate_limit_error,
    is_auth_error,
    validate_api_key,
)
from .backoff import ExponentialBackoff, BackoffConfig

__all__ = [
    # Exceptions
    "KeycycleError",
    "KeycycleKeyError",
    "NoAvailableKeyError",
    "KeyNotFoundError",
    "InvalidKeyError",
    "RateLimitError",
    "ConfigurationError",
    "MissingEnvironmentVariableError",
    "InvalidConfigurationError",
    # Utilities
    "get_key_suffix",
    "is_rate_limit_error",
    "is_auth_error",
    "validate_api_key",
    # Backoff
    "ExponentialBackoff",
    "BackoffConfig",
]
