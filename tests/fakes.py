"""Stand-ins for the Gemini client and for tools, shared by the tests."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from google.genai.interactions import Usage as ReportedUsage

from core import agent_loop


def reported_usage(input=None, output=None, thought=None, total=None) -> ReportedUsage:
    """Build the SDK's usage report; a count left out is None, as in a real response."""
    return ReportedUsage(
        total_input_tokens=input,
        total_output_tokens=output,
        total_thought_tokens=thought,
        total_tokens=total,
    )


@dataclass
class Turn:
    """One scripted model response: its steps and, optionally, its usage report.

    A turn without a report answers with `usage=None`, as the API may do.
    """

    steps: list[dict]
    usage: Optional[ReportedUsage] = None


class FakeStep:
    """Stands in for an SDK step: only `model_dump` is used by the agent."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self, **kwargs):
        return dict(self._fields)


class FakeInteractions:
    """Stands in for `client.aio.interactions`; records the calls it receives."""

    def __init__(self, steps, usage=None):
        self.calls = []
        self._interaction = SimpleNamespace(steps=steps, usage=usage)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._interaction


class ScriptedInteractions:
    """Stands in for `client.aio.interactions`; answers each call with the next scripted turn.

    Each turn is a list of step dicts, or a `Turn` when the response should also
    carry a usage report. Calling the model more often than the test scripted
    fails the test, so a runaway loop cannot hang it.
    """

    def __init__(self, *turns):
        self.calls = []
        self._turns = [turn if isinstance(turn, Turn) else Turn(turn) for turn in turns]

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > len(self._turns):
            raise AssertionError("the model was called more often than the test scripted")
        turn = self._turns[len(self.calls) - 1]
        steps = [FakeStep(**step) for step in turn.steps]
        return SimpleNamespace(steps=steps, usage=turn.usage)


def use_fake(monkeypatch, fake):
    """Make the agents talk to `fake` instead of the real Gemini client."""
    client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
    monkeypatch.setattr(agent_loop, "_get_client", lambda: client)


def user_step(text):
    """Build the user step the way run_turn records it."""
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def reply_step(text):
    """Build a model_output step as the model returns it."""
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def call_step(call_id, name, arguments):
    """Build a function_call step as the model returns it."""
    return {"type": "function_call", "id": call_id, "name": name, "arguments": arguments}


def result_of(history, call_id):
    """Return the function_result step answering `call_id`; fail unless there is exactly one."""
    results = [s for s in history if s["type"] == "function_result" and s["call_id"] == call_id]
    assert len(results) == 1
    return results[0]


def stub_tools(effects):
    """A small tool table that needs no shop, for tests of the generic agent loop.

    `lookup(key)` only reads. `record(key, amount=None)` has a side effect, which
    it appends to `effects`, so a test can tell whether a tool really ran.
    """
    async def lookup(key):
        return {"ok": True, "key": key, "value": f"value of {key}"}

    async def record(key, amount=None):
        effects.append((key, amount))
        return {"ok": True, "key": key, "amount": amount}

    return {"lookup": lookup, "record": record}
