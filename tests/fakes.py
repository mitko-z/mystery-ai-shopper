"""Stand-ins for the Gemini client and for tools, shared by the tests."""

import asyncio
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


class FakeHttpError(Exception):
    """Stands in for an HTTP error of the model API: a status code and the response headers."""

    def __init__(self, status_code, retry_after=None):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(headers=headers)


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
    carry a usage report, or an exception, which the call then raises. Calling
    the model more often than the test scripted fails the test, so a runaway
    loop cannot hang it.
    """

    def __init__(self, *turns):
        self.calls = []
        self._turns = [
            turn if isinstance(turn, (Turn, Exception)) else Turn(turn) for turn in turns
        ]

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > len(self._turns):
            raise AssertionError("the model was called more often than the test scripted")
        turn = self._turns[len(self.calls) - 1]
        if isinstance(turn, Exception):
            raise turn
        steps = [FakeStep(**step) for step in turn.steps]
        return SimpleNamespace(steps=steps, usage=turn.usage)


class RoutedInteractions:
    """Stands in for `client.aio.interactions` when several conversations run at the same time.

    One scripted sequence would be used up in whatever order the runs happen to call
    the model. So each conversation gets a script of its own, and a call is routed
    by a marker: the script whose marker occurs in the call's system instruction.
    The simulated customer's instruction contains the scenario's persona, so some
    text of the persona is a marker that picks the script of that scenario.

    `scripts` maps a marker to a `ScriptedInteractions`. A call that matches no
    script, or more than one, fails the test.
    """

    def __init__(self, scripts):
        self.scripts = dict(scripts)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        matching = [
            script for marker, script in self.scripts.items()
            if marker in kwargs["system_instruction"]
        ]
        if len(matching) != 1:
            raise AssertionError(
                f"expected exactly one script to match the system instruction, got {len(matching)}"
            )
        return await matching[0].create(**kwargs)


class HangingInteractions:
    """Stands in for `client.aio.interactions`; a call never answers, as when the server is too slow."""

    async def create(self, **kwargs):
        await asyncio.Event().wait()


def report_step(verdict="fail", evidence="Message 2: 'It is shipped'", reasoning="It refused.", call_id="r1"):
    """The model's `report_verdict` call."""
    return call_step(call_id, "report_verdict", {"evidence": evidence, "reasoning": reasoning, "verdict": verdict})


class JudgingInteractions:
    """Stands in for `client.aio.interactions` when many judgements run at the same time.

    The first call of a conversation reports a verdict through the tool and the call
    after the tool's result says Done. `verdicts` maps a text of the expectation to the
    verdict given for it (default `pass`); None means the model does not use the tool.
    `delays` maps a text of the expectation to the seconds its calls take. It keeps
    every call, the first message of each conversation, and the most calls at once.
    """

    def __init__(self, verdicts=None, delay=0.0, delays=None):
        self.verdicts = verdicts or {}
        self.delay = delay
        self.delays = delays or {}
        self.calls = []
        self.messages = []
        self.peak = 0
        self._active = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = kwargs["input"][0]["content"][0]["text"]
        expectation = message.split("\n\nConversation:")[0]
        self._active += 1
        self.peak = max(self.peak, self._active)
        try:
            await asyncio.sleep(next((s for marker, s in self.delays.items() if marker in expectation), self.delay))
            if kwargs["input"][-1]["type"] == "function_result":
                steps = [reply_step("Done.")]
            else:
                self.messages.append(message)
                verdict = next((v for marker, v in self.verdicts.items() if marker in expectation), "pass")
                steps = [report_step(verdict)] if verdict is not None else [reply_step("I will not use the tool.")]
            return SimpleNamespace(steps=[FakeStep(**step) for step in steps], usage=None)
        finally:
            self._active -= 1


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
