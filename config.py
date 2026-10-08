"""Application configuration read from environment variables."""

import os

API_KEY_ENV_VAR = "GEMINI_API_KEY"


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
            "Set it to your Anthropic API key and restart the terminal/IDE."
        )
    return key
