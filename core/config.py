"""Application configuration read from environment variables."""

import os

API_KEY_ENV_VAR = "GEMINI_API_KEY"

# The most model calls one run may make, unless the caller asks for another limit.
DEFAULT_CALLS_PER_RUN = 60

# A model call that fails with one of these HTTP statuses (rate limit, server
# trouble) is tried again, see core.agent_loop. Other failures are not retried.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MODEL_CALL_RETRIES = 3
RETRY_PAUSE_SECONDS = 2.0  # before the first retry; it doubles with each further one
RETRY_JITTER = 0.25  # up to this share of the pause is added at random
MAX_RETRY_AFTER_SECONDS = 30.0  # a longer Retry-After is a quota we cannot wait out


class MissingAPIKeyError(RuntimeError):
    """Raised when the GEMINI API key is not set in the environment."""


def get_api_key() -> str:
    """Return the GEMINI API key from the environment.

    Returns:
        The key with surrounding whitespace removed.

    Raises:
        MissingAPIKeyError: If the variable is unset or blank. The message
            never contains the key value.
    """
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"Environment variable {API_KEY_ENV_VAR} is not set. "
            "Set it to your Gemini API key and restart the terminal/IDE."
        )
    return key
