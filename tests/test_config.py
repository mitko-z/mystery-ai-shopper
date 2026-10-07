import pytest

import config


def test_get_api_key_returns_value(monkeypatch):
    """The key is read from the environment."""
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "test-key")
    assert config.get_api_key() == "test-key"


def test_get_api_key_strips_whitespace(monkeypatch):
    """Surrounding whitespace and newlines are removed."""
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "  test-key\n")
    assert config.get_api_key() == "test-key"


def test_get_api_key_missing(monkeypatch):
    """An unset variable raises a clear error."""
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    with pytest.raises(config.MissingAPIKeyError, match=config.API_KEY_ENV_VAR):
        config.get_api_key()


def test_get_api_key_blank(monkeypatch):
    """A blank variable is treated as missing."""
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "   ")
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()
