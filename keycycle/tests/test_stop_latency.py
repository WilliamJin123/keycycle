"""stop() must return promptly.

The cleanup thread used to sleep CLEANUP_INTERVAL_SECONDS (55s) between passes,
so stop() burned its whole 10s join timeout for every manager built — a test
suite that builds twenty managers paid minutes for nothing. The loop now waits
on the stop event, which stop() sets first.
"""

import time

from keycycle.config.enums import RateLimitStrategy
from keycycle.key_rotation.rotation_manager import RotatingKeyManager


def test_stop_returns_promptly_without_a_database():
    manager = RotatingKeyManager(
        api_keys=["sk-test-1-AAAAAAAA"],
        provider_name="stoplatency",
        strategy=RateLimitStrategy.PER_MODEL,
        track_usage=False,
    )
    assert manager._thread.is_alive()
    started = time.perf_counter()
    manager.stop()
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"stop() took {elapsed:.1f}s"
    assert not manager._thread.is_alive()
