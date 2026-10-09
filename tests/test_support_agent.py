import importlib
import sys
from types import SimpleNamespace

import pytest

import config

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture
def agent(monkeypatch):
    """Import a fresh support_agent with no API key set."""
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    sys.modules.pop("support_agent", None)
    return importlib.import_module("support_agent")


class FakeStep:
    """Stands in for an SDK step: only `model_dump` is used by the agent."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self, **kwargs):
        return dict(self._fields)


class FakeInteractions:
    """Stands in for `client.aio.interactions`; records the calls it receives."""

    def __init__(self, steps):
        self.calls = []
        self._interaction = SimpleNamespace(steps=steps)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._interaction


def use_fake(agent, monkeypatch, fake):
    """Make the agent talk to `fake` instead of the real Gemini client."""
    client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
    monkeypatch.setattr(agent, "_get_client", lambda: client)


def user_step(text):
    """Build the user step the way run_turn records it."""
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def reply_step(text):
    """Build a model_output step as the model returns it."""
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


async def test_import_without_api_key(agent):
    """Importing the module does not need the API key."""
    assert callable(agent.run_turn)


async def test_missing_key_fails_on_first_use(agent):
    """The missing-key error surfaces when the turn runs, not on import."""
    history = []
    with pytest.raises(config.MissingAPIKeyError):
        await agent.run_turn(history, "hi")
    assert history == []


async def test_turn_is_recorded_in_history_only(agent, monkeypatch):
    """The user message and the reply are appended, and nothing is returned."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Hello!"))])
    use_fake(agent, monkeypatch, fake)
    history = []

    result = await agent.run_turn(history, "hi")

    assert result is None
    assert history == [user_step("hi"), reply_step("Hello!")]


async def test_history_is_sent_before_the_new_message(agent, monkeypatch):
    """Earlier turns are passed to the model ahead of the new user message."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Sure."))])
    use_fake(agent, monkeypatch, fake)
    history = [user_step("hi"), reply_step("Hello!")]

    await agent.run_turn(history, "stock of P1?")

    sent = fake.calls[0]["input"]
    assert sent == [user_step("hi"), reply_step("Hello!"), user_step("stock of P1?")]
    assert history == [*sent, reply_step("Sure.")]


async def test_tool_calls_are_recorded_in_history(agent, monkeypatch, capsys):
    """Every step the model produced is kept, tool calls included, in order."""
    thought = {"type": "thought", "signature": "sig"}
    call = {"type": "function_call", "id": "call_1", "name": "check_stock",
            "arguments": {"product_id": "P1"}, "signature": "sig"}
    fake = FakeInteractions(steps=[FakeStep(**thought), FakeStep(**call)])
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.run_turn(history, "stock of P1?")

    assert history == [user_step("stock of P1?"), thought, call]
    assert "check_stock" in capsys.readouterr().out


async def test_history_untouched_when_response_has_no_steps(agent, monkeypatch):
    """A response without steps raises and leaves the history as it was."""
    use_fake(agent, monkeypatch, FakeInteractions(steps=None))
    history = [user_step("hi")]

    with pytest.raises(ValueError):
        await agent.run_turn(history, "again")

    assert history == [user_step("hi")]
