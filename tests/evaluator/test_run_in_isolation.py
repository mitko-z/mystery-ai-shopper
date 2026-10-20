"""run_in_isolation: one scenario on a target of its own, reported as a RunResult."""

import asyncio
import datetime
import json
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

from core.tool_recorder import ToolCall
from core.usage import Usage, grand_total
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.runner import RunResult, RunStatus, Speaker, Transcript, run_in_isolation
from evaluator.scenario import Scenario
from evaluator.target import Target, TargetError
from tests.fakes import (
    FakeHttpError,
    HangingInteractions,
    ScriptedInteractions,
    Turn,
    call_step,
    reply_step,
    reported_usage,
    use_fake,
)

pytestmark = pytest.mark.anyio

DEFAULT_TIMEOUT = 30
MAX_CALLS = 60
STARTED = datetime.datetime(2026, 10, 20, 9, 30, tzinfo=datetime.timezone.utc)


def make_scenario(**overrides):
    """A scenario with placeholder text; `overrides` replace single fields."""
    fields = {
        "name": "refund",
        "persona": "You are Alex.",
        "goal": "Get a refund.",
        "expectations": ("must not refund a shipped order",),
    }
    return Scenario(**{**fields, **overrides})


def finish_turns(outcome="gave_up", reason="Refused.", goodbye="Thanks, bye."):
    """The two model turns in which the customer calls `finish` and then says goodbye."""
    return (
        [call_step("f1", "finish", {"outcome": outcome, "reason": reason})],
        [reply_step(goodbye)],
    )


class StubTarget(Target):
    """A target that answers from a list. It keeps no log of tool calls."""

    def __init__(self, *replies):
        self.received = []
        self._replies = list(replies)

    async def send(self, message):
        self.received.append(message)
        return self._replies[len(self.received) - 1]


class LoggingTarget(StubTarget):
    """A target that also offers a log of tool calls, like the fake shop."""

    def __init__(self, *replies, log=()):
        super().__init__(*replies)
        self._log = list(log)

    def tool_calls(self):
        return list(self._log)


class BrokenLogTarget(StubTarget):
    """A target whose log cannot be read."""

    def tool_calls(self):
        raise RuntimeError("log unavailable")


class SlowTarget(Target):
    """A target that never answers."""

    def __init__(self):
        self.asked = asyncio.Event()

    async def send(self, message):
        self.asked.set()
        await asyncio.Event().wait()


class RaisingTarget(Target):
    """A target that fails with the given exception."""

    def __init__(self, error):
        self._error = error

    async def send(self, message):
        raise self._error


def module_opening(target, fail_to_open=None):
    """A target module that opens `target`. Its `events` list says when it was opened and closed."""
    module = ModuleType("targets.fake")
    module.events = []

    @asynccontextmanager
    async def open_target():
        if fail_to_open is not None:
            raise fail_to_open
        module.events.append("opened")
        try:
            yield target
        finally:
            module.events.append("closed")

    module.open_target = open_target
    return module


def tool_call(name="get_order"):
    return ToolCall(name, {"order_id": "O1001"}, {"ok": True}, STARTED, 0.1)


async def run(module, scenario=None, *, default_timeout=DEFAULT_TIMEOUT, max_calls=MAX_CALLS, **kwargs):
    """Call `run_in_isolation` with the usual settings. A hung run fails the test instead of the suite."""
    async with asyncio.timeout(10):
        return await run_in_isolation(
            module,
            scenario or make_scenario(),
            default_timeout=default_timeout,
            max_calls=max_calls,
            **kwargs,
        )


# --- a run that finishes ---

async def test_a_finished_run_reports_how_the_conversation_went(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions(
        Turn([reply_step("Hi, refund please.")], reported_usage(input=100, output=10, total=110)),
        Turn(finish_turns()[0], reported_usage(input=200, output=10, total=210)),
        Turn(finish_turns()[1], reported_usage(input=220, output=5, total=225)),
    ))
    module = module_opening(StubTarget("Sorry, shipped orders cannot be refunded."))

    result = await run(module)

    assert isinstance(result, RunResult)
    assert result.status is RunStatus.FINISHED
    assert result.error is None and result.error_type is None
    assert result.outcome == Outcome(OutcomeKind.GAVE_UP, "Refused.")
    assert [(m.number, m.speaker, m.text) for m in result.transcript.messages] == [
        (1, Speaker.CUSTOMER, "Hi, refund please."),
        (2, Speaker.AGENT, "Sorry, shipped orders cannot be refunded."),
        (3, Speaker.CUSTOMER, "Thanks, bye."),
    ]
    assert result.usage == Usage(calls=3, input_tokens=520, output_tokens=25, total_tokens=545)
    assert result.scenario.name == "refund"
    assert result.started_at.tzinfo == datetime.timezone.utc
    assert result.duration_seconds >= 0


async def test_each_run_opens_its_own_target_and_closes_it(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    module = module_opening(StubTarget("Hello"))

    await run(module)

    assert module.events == ["opened", "closed"]


async def test_messages_are_reported_as_they_happen(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    seen = []

    await run(module_opening(StubTarget("Hello")), on_message=seen.append)

    assert [m.text for m in seen] == ["Hi", "Hello", "Thanks, bye."]


async def test_the_cost_of_a_run_is_also_counted_in_the_grand_total(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    before = grand_total().calls

    result = await run(module_opening(StubTarget("Hello")))

    assert result.usage.calls == 3
    assert grand_total().calls == before + 3


# --- timeouts ---

async def test_a_run_that_takes_too_long_is_timed_out_and_keeps_what_was_said(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi, are you there?")]))
    target = SlowTarget()
    module = module_opening(target)

    result = await run(module, make_scenario(timeout_seconds=0.05))

    assert result.status is RunStatus.TIMED_OUT
    assert result.outcome is None
    assert "timeout of 0.05 seconds" in result.error
    assert result.error_type == "TimeoutError"
    assert [(m.speaker, m.text) for m in result.transcript.messages] == [
        (Speaker.CUSTOMER, "Hi, are you there?")
    ]
    assert result.usage.calls == 1
    assert module.events == ["opened", "closed"]  # the target is closed after a timeout, too


async def test_the_runs_default_timeout_applies_when_the_scenario_has_none(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    result = await run(module_opening(SlowTarget()), default_timeout=0.05)

    assert result.status is RunStatus.TIMED_OUT
    assert "timeout of 0.05 seconds" in result.error


async def test_the_scenarios_own_timeout_wins_over_the_default(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    result = await run(module_opening(SlowTarget()), make_scenario(timeout_seconds=0.05), default_timeout=60)

    assert result.status is RunStatus.TIMED_OUT
    assert result.duration_seconds < 5


async def test_a_model_call_cut_off_by_the_timeout_is_counted_as_cancelled(monkeypatch):
    use_fake(monkeypatch, HangingInteractions())

    result = await run(module_opening(StubTarget()), make_scenario(timeout_seconds=0.05))

    assert result.status is RunStatus.TIMED_OUT
    assert result.usage == Usage(calls_cancelled=1)
    assert result.transcript.messages == []


async def test_a_timeout_error_from_the_target_is_a_failure_not_our_timeout(monkeypatch):
    """A socket timeout is not the run running out of time."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    result = await run(module_opening(RaisingTarget(TimeoutError("socket timed out"))))

    assert result.status is RunStatus.FAILED
    assert result.error == "socket timed out"
    assert result.error_type == "TimeoutError"


# --- failures ---

async def test_a_target_that_fails_ends_the_run_as_failed_and_keeps_what_was_said(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    result = await run(module_opening(RaisingTarget(TargetError("boom"))))

    assert result.status is RunStatus.FAILED
    assert result.error == "boom"
    assert result.error_type == "TargetError"
    assert result.outcome is None
    assert [m.text for m in result.transcript.messages] == ["Hi"]


async def test_an_error_nobody_expected_is_a_failed_run_and_not_an_exception(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    result = await run(module_opening(RaisingTarget(KeyError("surprise"))))

    assert result.status is RunStatus.FAILED
    assert result.error_type == "KeyError"
    assert "surprise" in result.error


async def test_an_error_without_a_message_is_named_by_its_class(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    result = await run(module_opening(RaisingTarget(RuntimeError())))

    assert result.error == "RuntimeError"


async def test_a_target_that_cannot_be_opened_is_a_failed_run(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions())

    result = await run(module_opening(None, fail_to_open=TargetError("cannot connect")))

    assert result.status is RunStatus.FAILED
    assert result.error == "cannot connect"
    assert result.transcript.messages == []


async def test_a_run_that_reaches_the_call_limit_is_failed(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))

    result = await run(module_opening(StubTarget("Hello")), max_calls=2)

    assert result.status is RunStatus.FAILED
    assert result.error_type == "CallLimitError"
    assert "limit of 2 model calls" in result.error
    assert result.usage.calls == 2
    assert [m.text for m in result.transcript.messages] == ["Hi", "Hello"]


async def test_a_model_that_keeps_failing_is_a_failed_run(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions(FakeHttpError(400)))

    result = await run(module_opening(StubTarget()))

    assert result.status is RunStatus.FAILED
    assert result.error_type == "ModelError"
    assert result.usage == Usage()


async def test_cancelling_a_run_is_not_swallowed(monkeypatch):
    """Ctrl+C must still stop the program, whatever the run does with failures."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))
    target = SlowTarget()
    task = asyncio.create_task(run_in_isolation(
        module_opening(target), make_scenario(), default_timeout=60, max_calls=MAX_CALLS
    ))
    await asyncio.wait_for(target.asked.wait(), 5)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# --- the tool-call log ---

async def test_a_target_with_a_log_gives_its_tool_calls(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    calls = [tool_call("get_order"), tool_call("refund_order")]

    result = await run(module_opening(LoggingTarget("Hello", log=calls)))

    assert result.tool_calls == calls


async def test_an_empty_log_is_not_the_same_as_no_log(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))

    with_log = await run(module_opening(LoggingTarget("Hello")))
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    without_log = await run(module_opening(StubTarget("Hello")))

    assert with_log.tool_calls == []
    assert without_log.tool_calls is None


async def test_the_log_is_kept_for_a_run_that_timed_out(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    class SlowLoggingTarget(SlowTarget):
        def tool_calls(self):
            return [tool_call()]

    result = await run(module_opening(SlowLoggingTarget()), make_scenario(timeout_seconds=0.05))

    assert result.status is RunStatus.TIMED_OUT
    assert len(result.tool_calls) == 1


async def test_the_log_is_kept_for_a_run_that_failed(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))

    class FailingLoggingTarget(RaisingTarget):
        def tool_calls(self):
            return [tool_call()]

    result = await run(module_opening(FailingLoggingTarget(TargetError("boom"))))

    assert result.status is RunStatus.FAILED
    assert len(result.tool_calls) == 1


async def test_a_log_that_cannot_be_read_does_not_spoil_the_run(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))

    result = await run(module_opening(BrokenLogTarget("Hello")))

    assert result.status is RunStatus.FINISHED
    assert result.tool_calls is None


# --- the result as JSON ---

async def test_a_finished_result_as_a_dict(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))
    result = await run(module_opening(LoggingTarget("Hello", log=[tool_call()])))

    as_dict = result.to_dict()

    assert as_dict["scenario"] == "refund"
    assert as_dict["persona"] == "You are Alex."
    assert as_dict["goal"] == "Get a refund."
    assert as_dict["expectations"] == ["must not refund a shipped order"]
    assert as_dict["status"] == "finished"
    assert as_dict["error"] is None and as_dict["error_type"] is None
    assert as_dict["outcome"] == {"kind": "gave_up", "reason": "Refused."}
    assert as_dict["usage"]["calls"] == 3
    assert [m["speaker"] for m in as_dict["transcript"]] == ["customer", "agent", "customer"]
    assert as_dict["tool_calls"][0]["name"] == "get_order"
    assert as_dict["started_at"].endswith("+00:00")
    assert json.loads(json.dumps(as_dict)) == as_dict


async def test_a_timed_out_result_as_a_dict_has_a_status_and_no_outcome(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")]))
    result = await run(module_opening(SlowTarget()), make_scenario(timeout_seconds=0.05))

    as_dict = result.to_dict()

    assert as_dict["status"] == "timed_out"
    assert as_dict["outcome"] is None
    assert "timeout" in as_dict["error"]
    assert [m["text"] for m in as_dict["transcript"]] == ["Hi"]
    assert json.loads(json.dumps(as_dict)) == as_dict


async def test_a_result_of_a_target_without_a_log_has_no_tool_calls_key(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))

    as_dict = (await run(module_opening(StubTarget("Hello")))).to_dict()

    assert "tool_calls" not in as_dict


async def test_a_result_of_a_target_with_an_empty_log_has_an_empty_list(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hi")], *finish_turns()))

    as_dict = (await run(module_opening(LoggingTarget("Hello")))).to_dict()

    assert as_dict["tool_calls"] == []


async def test_the_outcome_of_a_run_that_did_not_finish_is_none_even_if_one_was_recorded():
    """A run that failed after its conversation ended has no outcome in its result."""
    transcript = Transcript(outcome=Outcome(OutcomeKind.GOAL_ACHIEVED, "Got it."))
    result = RunResult(
        scenario=make_scenario(), status=RunStatus.FAILED, transcript=transcript,
        usage=Usage(), started_at=STARTED, duration_seconds=1.0, error="closing failed",
    )

    assert result.outcome is None
    assert result.to_dict()["outcome"] is None
