"""
API Key Management Utilities
"""
import os
import random
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ApiKeyManager:
    """Manages API keys with rotation and rate limit handling."""

    def __init__(self, keys: List[str] = None, env_var_name: str = None):
        """Initialize with keys from environment or provided list."""
        self.keys = keys or []
        self.current_index = 0
        self.last_used = {}
        self.min_delay = 1  # Minimum delay between uses of the same key (seconds)
        self.failed_keys = {}  # Track keys that have failed with their failure count
        self.max_failures = 5  # Maximum failures before temporarily disabling a key
        self.failure_timeout = 300  # Time to keep a key disabled after max failures (seconds)

        # Load keys from environment if not provided
        if not self.keys and env_var_name:
            # Try to get keys from environment variable as comma-separated string
            env_keys = os.getenv(env_var_name, "")
            if env_keys:
                self.keys = [k.strip() for k in env_keys.split(",") if k.strip()]

        if not self.keys:
            raise ValueError(f"No API keys available. Please provide keys via environment variables or constructor.")

    def get_next_key(self) -> str:
        """Get the next available API key with rate limiting and failure handling."""
        if not self.keys:
            raise ValueError("No API keys available")

        # Clean up any expired failed keys
        self._cleanup_failed_keys()

        # Get list of currently available keys (not disabled due to failures)
        available_keys = [k for k in self.keys if self._is_key_available(k)]

        if not available_keys:
            # If all keys are disabled, reset the one that failed longest ago
            oldest_failure = float('inf')
            oldest_key = None

            for key, data in self.failed_keys.items():
                if data["timestamp"] < oldest_failure:
                    oldest_failure = data["timestamp"]
                    oldest_key = key

            if oldest_key:
                print(f"All keys are disabled. Resetting the oldest failed key.")
                self.failed_keys.pop(oldest_key, None)
                available_keys = [oldest_key]
            else:
                # Fallback to all keys if something went wrong with tracking
                available_keys = self.keys

        # Try each available key until we find one that's not rate-limited
        attempts = 0
        max_attempts = len(available_keys) * 2  # Allow multiple passes through the keys

        while attempts < max_attempts:
            # Get next key with round-robin rotation
            if not available_keys:
                # If we somehow have no available keys, wait and retry
                print("No keys available. Waiting 5 seconds before retrying...")
                time.sleep(5)
                return self.get_next_key()

            # Use modulo to cycle through available keys
            idx = attempts % len(available_keys)
            key = available_keys[idx]

            # Check if this key was recently used
            last_use = self.last_used.get(key, 0)
            time_since_last_use = time.time() - last_use

            if time_since_last_use >= self.min_delay:
                self.last_used[key] = time.time()
                return key

            # If key was recently used but this is our last option, wait
            if attempts >= len(available_keys) - 1:
                sleep_time = max(0.1, self.min_delay - time_since_last_use)
                print(f"All keys are rate-limited. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                self.last_used[key] = time.time()
                return key

            attempts += 1

        # Fallback to random key if rotation fails
        key = random.choice(available_keys if available_keys else self.keys)
        self.last_used[key] = time.time()
        return key

    def _cleanup_failed_keys(self):
        """Remove keys from the failed list if their timeout has expired."""
        current_time = time.time()
        keys_to_remove = []

        for key, data in self.failed_keys.items():
            if current_time - data["timestamp"] > self.failure_timeout:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            print(f"Re-enabling previously failed API key.")
            self.failed_keys.pop(key, None)

    def _is_key_available(self, key):
        """Check if a key is available (not disabled due to failures)."""
        if key not in self.failed_keys:
            return True

        # Check if key has reached max failures and is still within timeout
        data = self.failed_keys[key]
        if data["count"] >= self.max_failures:
            time_since_failure = time.time() - data["timestamp"]
            if time_since_failure <= self.failure_timeout:
                return False

        return True

    def report_failure(self, key):
        """Report a key failure to potentially disable it temporarily."""
        if key not in self.failed_keys:
            self.failed_keys[key] = {"count": 1, "timestamp": time.time()}
        else:
            self.failed_keys[key]["count"] += 1
            self.failed_keys[key]["timestamp"] = time.time()

            if self.failed_keys[key]["count"] >= self.max_failures:
                print(f"API key has failed {self.failed_keys[key]['count']} times and will be temporarily disabled.")

    def report_success(self, key):
        """Report a successful API call to reset failure count."""
        if key in self.failed_keys:
            # Reduce failure count on success, but don't remove completely
            self.failed_keys[key]["count"] = max(0, self.failed_keys[key]["count"] - 1)
