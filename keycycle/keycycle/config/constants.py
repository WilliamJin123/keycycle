"""Timing constants for key rotation and rate limiting."""

# Time windows (in seconds)
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

# Cooldown configuration
DEFAULT_COOLDOWN_SECONDS = 30

# Cleanup intervals
CLEANUP_INTERVAL_SECONDS = 55

# Key wait/polling configuration
DEFAULT_POLL_INTERVAL = 0.5
MIN_POLL_INTERVAL = 0.1
MAX_POLL_INTERVAL = 5.0

# History lookback
HISTORY_LOOKBACK_SECONDS = 86400  # 24 hours

# API key suffix length for logging
KEY_SUFFIX_LENGTH = 8

# Temporary rate limit retry configuration
TEMP_RATE_LIMIT_MAX_RETRIES = 3
TEMP_RATE_LIMIT_INITIAL_DELAY = 1.0  # seconds
TEMP_RATE_LIMIT_MAX_DELAY = 10.0  # seconds
TEMP_RATE_LIMIT_MULTIPLIER = 2.0

# Key rotation delay (seconds to wait after rotating to a new key)
KEY_ROTATION_DELAY_SECONDS = 0.5

# Usage logging batch size
USAGE_LOG_BATCH_SIZE = 50
