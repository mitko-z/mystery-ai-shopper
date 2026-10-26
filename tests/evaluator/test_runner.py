import asyncio
import json

import pytest

from evaluator.customer import CustomerError
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.runner import Message, Speaker, Transcript, run_scenario
from evaluator.scenario import Scenario
from evaluator.target import Target, TargetError
from tests.fakes import (
    ScriptedInteractions,
    call_step,
    reply_step,
    use_fake,
)

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


async def play(target, scenario=None, **kwargs):
    """Run `scenario` against `target` the way a caller does: with a transcript of its own."""
    transcript = Transcript()
    await run_scenario(target, scenario or make_scenario(), transcript, **kwargs)
    return transcript


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

    transcript = await play(target)

    assert [(m.number, m.speaker, m.text) for m in transcript.messages] == [
        (1, Speaker.CUSTOMER, "Hi, refund for O1002 please."),
        (2, Speaker.AGENT, "Sorry, shipped orders cannot be refunded."),
        (3, Speaker.CUSTOMER, "Thanks, bye."),
    ]
    assert transcript.outcome == Outcome(OutcomeKind.GAVE_UP, "Refused.")
    assert target.received == ["Hi, refund for O1002 please."]


async def test_the_outcome_is_returned_and_stored_in_the_transcript(monkeypatch):
    """Both ways to get at the outcome give the same one."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    transcript = Transcript()

    outcome = await run_scenario(StubTarget("Hello"), make_scenario(), transcript)

    assert outcome is transcript.outcome
    assert outcome == Outcome(OutcomeKind.GAVE_UP, "Refused.")


async def test_customer_that_finishes_at_once_sends_nothing(monkeypatch):
    """A customer who leaves on the first turn never reaches the target."""
    use_fake(monkeypatch, ScriptedInteractions(*finish_turns(goodbye="Never mind.")))
    target = StubTarget()

    transcript = await play(target)

    assert target.received == []
    assert [m.text for m in transcript.messages] == ["Never mind."]


async def test_conversation_is_cut_off_at_the_message_limit(monkeypatch):
    """After max_messages messages the run ends as cut_off, and the extra message is not sent."""
    fake = ScriptedInteractions([reply_step("one")], [reply_step("two")], [reply_step("three")])
    use_fake(monkeypatch, fake)
    target = StubTarget("r1", "r2")

    transcript = await play(target, make_scenario(max_messages=2))

    assert target.received == ["one", "two"]
    assert [m.text for m in transcript.messages] == ["one", "r1", "two", "r2"]
    assert transcript.outcome.kind is OutcomeKind.CUT_OFF
    assert "2 messages" in transcript.outcome.reason
    assert len(fake.calls) == 3  # the customer got a last turn after the last answer


async def test_customer_can_finish_after_the_last_answer(monkeypatch):
    """A run that ends well on the very last answer is not reported as cut off."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns("goal_achieved", "Got it.")))

    transcript = await play(StubTarget("Done, refunded."), make_scenario(max_messages=1))

    assert transcript.outcome == Outcome(OutcomeKind.GOAL_ACHIEVED, "Got it.")


async def test_messages_are_reported_as_they_happen(monkeypatch):
    """`on_message` sees each message, in order, as it is added to the transcript."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    seen = []

    transcript = await play(StubTarget("Hello"), on_message=seen.append)

    assert seen == transcript.messages
    assert [m.number for m in seen] == [1, 2, 3]


async def test_a_failing_target_stops_the_run(monkeypatch):
    """A target error is not swallowed."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    with pytest.raises(TargetError, match="boom"):
        await play(FailingTarget())


# --- a transcript the caller owns ---

async def test_what_was_said_before_a_target_failure_is_kept(monkeypatch):
    """The caller's transcript still holds the opening message when the target fails, and has no outcome."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))
    transcript = Transcript()

    with pytest.raises(TargetError):
        await run_scenario(FailingTarget(), make_scenario(), transcript)

    assert [(m.speaker, m.text) for m in transcript.messages] == [(Speaker.CUSTOMER, "Hi")]
    assert transcript.outcome is None


async def test_what_was_said_before_a_timeout_is_kept(monkeypatch):
    """A run that is cancelled by a timeout leaves the conversation so far in the caller's transcript."""
    class SlowTarget(Target):
        async def send(self, message):
            await asyncio.sleep(10)

    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))
    transcript = Transcript()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(run_scenario(SlowTarget(), make_scenario(), transcript), timeout=0.05)

    assert [m.text for m in transcript.messages] == ["Hi"]
    assert transcript.outcome is None


async def test_what_was_said_before_a_customer_failure_is_kept(monkeypatch):
    """If the simulated customer fails on its second turn, the first exchange is still in the transcript."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], [reply_step("   ")]))
    transcript = Transcript()

    with pytest.raises(CustomerError):
        await run_scenario(StubTarget("Hello"), make_scenario(), transcript)

    assert [m.text for m in transcript.messages] == ["Hi", "Hello"]
    assert transcript.outcome is None


# --- the JSON-safe form ---

@pytest.mark.parametrize(
    "speaker, key", [(Speaker.CUSTOMER, "customer"), (Speaker.AGENT, "agent")]
)
def test_a_message_becomes_a_plain_dict(speaker, key):
    """The speaker is a lowercase key, not the text used when the conversation is printed."""
    as_dict = Message(3, speaker, "Hello").to_dict()

    assert as_dict == {"number": 3, "speaker": key, "text": "Hello"}
    assert type(as_dict["speaker"]) is str
    assert json.loads(json.dumps(as_dict)) == as_dict
