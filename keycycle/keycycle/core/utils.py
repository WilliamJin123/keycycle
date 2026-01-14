"""Shared utility functions for keycycle."""
from typing import FrozenSet

from ..config.constants import KEY_SUFFIX_LENGTH


# Rate limit indicators for detection
RATE_LIMIT_INDICATORS: FrozenSet[str] = frozenset([
    "429", "too many requests", "rate limit",
    "resource exhausted", "traffic", "rate-limited"
])

# Temporary/transient rate limit indicators - retry with SAME key
TEMPORARY_RATE_LIMIT_INDICATORS: FrozenSet[str] = frozenset([
    "temporarily rate-limited",
    "temporarily unavailable",
    "high traffic",
    "retry shortly",
    "please retry",
    "try again shortly",
    "rate-limited upstream",
    "experiencing high load",
])

# Hard/quota-based rate limit indicators - rotate to different key
HARD_RATE_LIMIT_INDICATORS: FrozenSet[str] = frozenset([
    "per-day",
    "per-hour",
    "per-minute",
    "quota exceeded",
    "daily limit",
    "hourly limit",
    "free-models-per-day",
    "x-ratelimit-remaining: 0",
])

# Auth error indicators
AUTH_STATUS_CODES = {401, 403}
AUTH_INDICATORS: FrozenSet[str] = frozenset([
    "401", "403", "unauthorized", "forbidden",
    "invalid api key", "invalid_api_key", "expired"
])


def get_key_suffix(api_key: str, length: int = KEY_SUFFIX_LENGTH) -> str:
    """
    Extract the suffix of an API key for logging and identification.

    Args:
        api_key: The full API key
        length: Number of characters to extract (default: 8)

    Returns:
        The last `length` characters, or the full key if shorter
    """
    return api_key[-length:] if len(api_key) > length else api_key


def is_rate_limit_error(e: Exception) -> bool:
    """
    Heuristic to detect rate limit errors across different providers.

    Checks:
        1. Exception message for rate limit keywords
        2. status_code attribute for 429
        3. body/response attribute for rate limit indicators

    Args:
        e: The exception to check

    Returns:
        True if this appears to be a rate limit error
    """
    # Check string representation
    err_str = str(e).lower()
    if any(indicator in err_str for indicator in RATE_LIMIT_INDICATORS):
        return True

    # Check status_code attribute
    if hasattr(e, "status_code") and e.status_code == 429:
        return True

    # Check body/response for embedded error info
    body = getattr(e, "body", None) or getattr(e, "response", None)
    if body:
        body_str = str(body).lower()
        if any(indicator in body_str for indicator in RATE_LIMIT_INDICATORS):
            return True

    return False


def is_temporary_rate_limit_error(e: Exception) -> bool:
    """
    Detect if a rate limit error is temporary (upstream congestion)
    vs. a hard quota limit.

    Temporary errors should be retried with the SAME key after a delay.
    Hard limit errors should trigger key rotation.

    Args:
        e: The exception to check

    Returns:
        True if this is a temporary rate limit that should be retried
        with the same key (no rotation needed)
    """
    # First, must be a rate limit error at all
    if not is_rate_limit_error(e):
        return False

    # Build combined error string from all available sources
    err_str = str(e).lower()

    body = getattr(e, "body", None) or getattr(e, "response", None)
    if body:
        err_str += " " + str(body).lower()

    # Check for hard limit indicators first (these take precedence)
    if any(indicator in err_str for indicator in HARD_RATE_LIMIT_INDICATORS):
        return False

    # Check for temporary indicators
    if any(indicator in err_str for indicator in TEMPORARY_RATE_LIMIT_INDICATORS):
        return True

    return False


def is_auth_error(e: Exception) -> bool:
    """
    Detect authentication/authorization errors (invalid/expired keys).

    Args:
        e: The exception to check

    Returns:
        True if this appears to be an auth error (401/403)
    """
    # Check status_code
    if hasattr(e, "status_code") and e.status_code in AUTH_STATUS_CODES:
        return True

    # Check string representation
    err_str = str(e).lower()
    if any(indicator in err_str for indicator in AUTH_INDICATORS):
        return True

    return False


def validate_api_key(api_key: str) -> bool:
    """
    Basic validation of API key format.

    Args:
        api_key: The API key to validate

    Returns:
        True if the key appears to be valid format

    Note:
        This only checks format, not whether the key actually works.
    """
    if not api_key or not isinstance(api_key, str):
        return False

    # Most API keys are at least 20 characters
    if len(api_key) < 20:
        return False

    # Check for common placeholder patterns (exact matches or obvious placeholders)
    placeholder_patterns = ["your_api_key", "placeholder", "your-api-key", "insert_key", "api_key_here"]
    key_lower = api_key.lower()
    if any(pattern in key_lower for pattern in placeholder_patterns):
        return False

    # Check if key is all x's or looks like a template
    if key_lower.replace("-", "").replace("_", "").replace("sk", "") == "x" * (len(api_key) - api_key.count("-") - api_key.count("_") - api_key.count("sk")):
        return False

    return True
