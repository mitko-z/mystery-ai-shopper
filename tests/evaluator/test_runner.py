import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

from core import agent_loop, config
from core.usage import Usage
from evaluator import runner
from evaluator.customer import CustomerError
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.runner import Message, Speaker, Transcript, run_scenario
from evaluator.scenario import Scenario
from evaluator.target import Target, TargetError
from tests.fakes import (
    FakeHttpError,
    ScriptedInteractions,
    Turn,
    call_step,
    reply_step,
    reported_usage,
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


# --- what a run cost ---

class ModelBackedTarget(Target):
    """A target that answers with a model call of its own, as the shop's support agent does."""

    async def send(self, message):
        history = []
        spec = agent_loop.AgentSpec(system_instruction="you are support", tool_schemas=[])
        await agent_loop.get_complete_reply({}, history, message, spec)
        return agent_loop.final_text(history)


def test_format_usage_names_calls_and_tokens():
    usage = Usage(calls=7, input_tokens=5120, output_tokens=410, thought_tokens=880, total_tokens=6410)

    assert runner.format_usage(usage) == (
        "Model calls: 7 | Tokens: input 5,120, output 410, thought 880, total 6,410"
    )


def test_format_usage_names_calls_that_reported_no_usage():
    usage = Usage(calls=7, input_tokens=100, total_tokens=100, calls_without_usage=2)

    assert runner.format_usage(usage).endswith(
        " | Calls without a usage report: 2 (tokens are incomplete)"
    )


def test_format_usage_of_an_unused_run():
    assert runner.format_usage(Usage()) == (
        "Model calls: 0 | Tokens: input 0, output 0, thought 0, total 0"
    )


def test_main_prints_what_the_run_cost_counting_the_customer_and_the_target(monkeypatch, capsys):
    """Calls of the customer and of the target's own model are summed in one line."""
    finish_call, goodbye = finish_turns()
    use_fake(monkeypatch, ScriptedInteractions(
        Turn([reply_step("Hi")], reported_usage(input=100, output=10, total=110)),
        Turn([reply_step("Hello")], reported_usage(input=50, output=5, thought=5, total=60)),
        Turn(finish_call, reported_usage(input=200, output=10, total=210)),
        Turn(goodbye, reported_usage(input=220, output=5, total=225)),
    ))
    fake_target_module(monkeypatch, ModelBackedTarget())

    code = runner.main(["fake", "refund"])

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines[-2] == "Outcome: gave_up - Refused."
    assert lines[-1] == "Model calls: 4 | Tokens: input 570, output 30, thought 5, total 605"


def test_main_still_prints_what_the_run_cost_when_it_fails(monkeypatch, capsys):
    """The calls made before a failure cost money too, so they are reported."""
    use_fake(monkeypatch, ScriptedInteractions(
        Turn([reply_step("Hi")], reported_usage(input=100, output=10, total=110)),
    ))
    fake_target_module(monkeypatch, FailingTarget())

    code = runner.main(["fake", "refund"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.splitlines()[-1] == (
        "Model calls: 1 | Tokens: input 100, output 10, thought 0, total 110"
    )
    assert "Outcome:" not in captured.out
    assert captured.err.strip() == "error: boom"


def test_main_prints_no_cost_for_a_run_that_never_started(capsys):
    """A mistyped target name is a usage error, not a run."""
    runner.main(["nosuch", "refund"])

    assert "Model calls" not in capsys.readouterr().out


def test_format_usage_names_cancelled_calls():
    usage = Usage(calls=3, input_tokens=100, total_tokens=100, calls_cancelled=1)

    assert runner.format_usage(usage).endswith(" | Calls cancelled (tokens unknown): 1")


# --- the call limit ---

def test_main_stops_a_run_that_reaches_the_call_limit(monkeypatch, capsys):
    """The run ends as an error; what was said and what it cost are still shown."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    fake_target_module(monkeypatch, StubTarget("Hello"))

    code = runner.main(["fake", "refund", "--max-calls", "2"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "error: This run reached its limit of 2 model calls."
    assert "1. Customer: Hi" in captured.out
    assert "2. Agent: Hello" in captured.out
    assert "Model calls: 2" in captured.out
    assert "Outcome:" not in captured.out


@pytest.mark.parametrize("value", ["0", "-3"])
def test_main_rejects_a_call_limit_below_one(value, capsys):
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["fake", "refund", "--max-calls", value])

    assert exit_info.value.code == 2
    assert "--max-calls must be at least 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra_args, expected",
    [([], config.DEFAULT_CALLS_PER_RUN), (["--max-calls", "7"], 7)],
)
def test_main_gives_the_run_its_call_limit(extra_args, expected, monkeypatch):
    seen = {}

    async def fake_run(target_name, scenario_name, max_calls):
        seen["max_calls"] = max_calls

    monkeypatch.setattr(runner, "_run", fake_run)

    runner.main(["fake", "refund", *extra_args])

    assert seen == {"max_calls": expected}


# --- a model that fails ---

def test_main_reports_a_failing_model_in_one_line(monkeypatch, capsys):
    """When the model keeps failing, the run ends with a short message, not a traceback."""
    use_fake(monkeypatch, ScriptedInteractions(*[FakeHttpError(503) for _ in range(4)]))
    fake_target_module(monkeypatch, StubTarget("Hello"))

    code = runner.main(["fake", "refund"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: The model call failed:")
    assert "Model calls: 0" in captured.out


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
