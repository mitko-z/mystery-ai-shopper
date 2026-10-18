import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

from evaluator import runner
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.runner import Speaker, run_scenario
from evaluator.scenario import Scenario
from evaluator.target import Target, TargetError
from tests.fakes import ScriptedInteractions, call_step, reply_step, use_fake

pytestmark = pytest.mark.anyio


def make_scenario(**overrides):
    """A scenario with placeholder text; `overrides` replace single fields."""
    fields = {"name": "refund", "persona": "You are Alex.", "goal": "Get a refund."}
    return Scenario(**{**fields, **overrides})


class StubTarget(Target):
    """A target that answers from a list and remembers what it was sent."""

    def __init__(self, *replies):
        self.received = []
        self._replies = list(replies)

    async def send(self, message):
        self.received.append(message)
        return self._replies[len(self.received) - 1]


class FailingTarget(Target):
    """A target that cannot answer."""

    async def send(self, message):
        raise TargetError("boom")


def finish_turns(outcome="gave_up", reason="Refused.", goodbye="Thanks, bye."):
    """The two model turns in which the customer calls `finish` and then says goodbye."""
    return (
        [call_step("f1", "finish", {"outcome": outcome, "reason": reason})],
        [reply_step(goodbye)],
    )


# --- run_scenario ---

async def test_conversation_ends_when_the_customer_finishes(monkeypatch):
    """The customer's goodbye is in the transcript but is never sent to the target."""
    fake = ScriptedInteractions([reply_step("Hi, refund for O1002 please.")], *finish_turns())
    use_fake(monkeypatch, fake)
    target = StubTarget("Sorry, shipped orders cannot be refunded.")

    transcript = await run_scenario(target, make_scenario())

    assert [(m.number, m.speaker, m.text) for m in transcript.messages] == [
        (1, Speaker.CUSTOMER, "Hi, refund for O1002 please."),
        (2, Speaker.AGENT, "Sorry, shipped orders cannot be refunded."),
        (3, Speaker.CUSTOMER, "Thanks, bye."),
    ]
    assert transcript.outcome == Outcome(OutcomeKind.GAVE_UP, "Refused.")
    assert target.received == ["Hi, refund for O1002 please."]


async def test_customer_that_finishes_at_once_sends_nothing(monkeypatch):
    """A customer who leaves on the first turn never reaches the target."""
    use_fake(monkeypatch, ScriptedInteractions(*finish_turns(goodbye="Never mind.")))
    target = StubTarget()

    transcript = await run_scenario(target, make_scenario())

    assert target.received == []
    assert [m.text for m in transcript.messages] == ["Never mind."]


async def test_conversation_is_cut_off_at_the_message_limit(monkeypatch):
    """After max_messages messages the run ends as cut_off, and the extra message is not sent."""
    fake = ScriptedInteractions([reply_step("one")], [reply_step("two")], [reply_step("three")])
    use_fake(monkeypatch, fake)
    target = StubTarget("r1", "r2")

    transcript = await run_scenario(target, make_scenario(max_messages=2))

    assert target.received == ["one", "two"]
    assert [m.text for m in transcript.messages] == ["one", "r1", "two", "r2"]
    assert transcript.outcome.kind is OutcomeKind.CUT_OFF
    assert "2 messages" in transcript.outcome.reason
    assert len(fake.calls) == 3  # the customer got a last turn after the last answer


async def test_customer_can_finish_after_the_last_answer(monkeypatch):
    """A run that ends well on the very last answer is not reported as cut off."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns("goal_achieved", "Got it.")))

    transcript = await run_scenario(StubTarget("Done, refunded."), make_scenario(max_messages=1))

    assert transcript.outcome == Outcome(OutcomeKind.GOAL_ACHIEVED, "Got it.")


async def test_messages_are_reported_as_they_happen(monkeypatch):
    """`on_message` sees each message, in order, as it is added to the transcript."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    seen = []

    transcript = await run_scenario(StubTarget("Hello"), make_scenario(), on_message=seen.append)

    assert seen == transcript.messages
    assert [m.number for m in seen] == [1, 2, 3]


async def test_a_failing_target_stops_the_run(monkeypatch):
    """A target error is not swallowed."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    with pytest.raises(TargetError, match="boom"):
        await run_scenario(FailingTarget(), make_scenario())


# --- command line ---

def fake_target_module(monkeypatch, target, scenarios=None):
    """Register a target module named `targets.fake` that opens `target`."""
    module = ModuleType("targets.fake")

    @asynccontextmanager
    async def open_target():
        yield target

    module.open_target = open_target
    module.SCENARIOS = scenarios if scenarios is not None else {"refund": make_scenario()}
    monkeypatch.setitem(sys.modules, "targets.fake", module)


def test_main_prints_the_conversation(monkeypatch, capsys):
    """A run prints the header, the numbered conversation and the outcome, and exits with 0."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    fake_target_module(monkeypatch, StubTarget("Hello"))

    code = runner.main(["fake", "refund"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Scenario: refund" in out
    assert "Persona: You are Alex." in out
    assert "Secret goal: Get a refund." in out
    assert "1. Customer: Hi" in out
    assert "2. Agent: Hello" in out
    assert "3. Customer: Thanks, bye." in out
    assert "Outcome: gave_up - Refused." in out


def test_main_names_the_available_targets_when_given_an_unknown_one(capsys):
    """A typo in the target name lists what exists, and exits with 1."""
    code = runner.main(["nosuch", "refund"])

    err = capsys.readouterr().err
    assert code == 1
    assert "Unknown target 'nosuch'" in err
    assert "shop" in err


def test_main_names_the_available_scenarios_when_given_an_unknown_one(monkeypatch, capsys):
    """A typo in the scenario name lists what the target has."""
    fake_target_module(monkeypatch, StubTarget())

    code = runner.main(["fake", "nosuch"])

    err = capsys.readouterr().err
    assert code == 1
    assert "Unknown scenario 'nosuch'" in err
    assert "refund" in err


def test_main_reports_a_target_failure_in_one_line(monkeypatch, capsys):
    """A failing target ends the run with a short message instead of a traceback."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))
    fake_target_module(monkeypatch, FailingTarget())

    code = runner.main(["fake", "refund"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "error: boom"
    assert "1. Customer: Hi" in captured.out  # what was said before the failure is still shown
