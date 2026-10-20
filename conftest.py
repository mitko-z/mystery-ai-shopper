import pytest

from core import agent_loop


@pytest.fixture
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
def no_waiting_between_retries(monkeypatch):
    """A failed model call is retried after a pause. Tests must not wait for it."""

    async def no_pause(seconds):
        pass

    monkeypatch.setattr(agent_loop, "_pause", no_pause)
