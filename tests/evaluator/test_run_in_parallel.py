"""run_in_parallel: many scenarios at once, with a cap on how many."""

import asyncio
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

from core.usage import grand_total
from evaluator.runner import RunResult, RunStatus, run_in_parallel
from evaluator.target import Target, TargetError
from tests.evaluator.helpers import make_scenarios, use_scripted_customers

pytestmark = pytest.mark.anyio


# --- targets that can be watched ---

class Gauge:
    """Counts how many runs are inside a target's `send` at the same moment."""

    def __init__(self):
        self.inside = 0
        self.peak = 0

    def enter(self):
        self.inside += 1
        self.peak = max(self.peak, self.inside)

    def leave(self):
        self.inside -= 1


class ProbeTarget(Target):
    """A target that reports to a gauge, and answers by doing `act`."""

    def __init__(self, gauge, act):
        self._gauge = gauge
        self._act = act

    async def send(self, message):
        self._gauge.enter()
        try:
            return await self._act(message)
        finally:
            self._gauge.leave()


def module_of(gauge, act):
    """A target module whose runs each get a ProbeTarget. It counts targets opened and closed."""
    module = ModuleType("targets.fake")
    module.opened = 0
    module.closed = 0

    @asynccontextmanager
    async def open_target():
        module.opened += 1
        try:
            yield ProbeTarget(gauge, act)
        finally:
            module.closed += 1

    module.open_target = open_target
    return module


def after(seconds):
    """An act that takes `seconds` to answer."""
    async def act(message):
        await asyncio.sleep(seconds)
        return "Hello"
    return act


def waits_for_others(gauge, count):
    """An act that answers only once `count` runs are inside `send` together."""
    everyone_here = asyncio.Event()

    async def act(message):
        if gauge.inside >= count:
            everyone_here.set()
        await asyncio.wait_for(everyone_here.wait(), 5)
        return "Hello"
    return act


async def run(module, scenarios, **kwargs):
    """Call `run_in_parallel` with the usual settings. A hung run fails the test instead of the suite."""
    settings = {"max_parallel": 5, "default_timeout": 30, "max_calls": 60, **kwargs}
    async with asyncio.timeout(15):
        return await run_in_parallel(module, scenarios, **settings)


# --- the results ---

async def test_every_scenario_is_run_and_the_results_follow_the_scenario_order(monkeypatch):
    use_scripted_customers(monkeypatch, 5)
    scenarios = make_scenarios(5)

    results = await run(module_of(Gauge(), after(0)), scenarios)

    assert all(isinstance(result, RunResult) for result in results)
    assert [result.scenario.name for result in results] == ["s0", "s1", "s2", "s3", "s4"]
    assert [result.status for result in results] == [RunStatus.FINISHED] * 5
    # each run had the conversation of its own scenario
    assert [result.transcript.messages[0].text for result in results] == [f"Hi from {i}" for i in range(5)]


async def test_results_keep_scenario_order_and_not_finish_order(monkeypatch):
    """The first scenario is the slowest, so it finishes last but is still first in the results."""
    use_scripted_customers(monkeypatch, 4)
    delays = {"Hi from 0": 0.15, "Hi from 1": 0.01, "Hi from 2": 0.01, "Hi from 3": 0.01}

    async def act(message):
        await asyncio.sleep(delays[message])
        return "Hello"

    finished = []
    results = await run(
        module_of(Gauge(), act), make_scenarios(4), on_finish=lambda result: finished.append(result.scenario.name)
    )

    assert [result.scenario.name for result in results] == ["s0", "s1", "s2", "s3"]
    assert finished[-1] == "s0"
    assert finished != ["s0", "s1", "s2", "s3"]


async def test_no_scenarios_give_no_results():
    assert await run(module_of(Gauge(), after(0)), []) == []


# --- the cap ---

async def test_scenarios_really_run_together_up_to_the_cap(monkeypatch):
    """With room for all five, all five are inside `send` at the same moment."""
    use_scripted_customers(monkeypatch, 5)
    gauge = Gauge()

    results = await run(module_of(gauge, waits_for_others(gauge, 5)), make_scenarios(5), max_parallel=5)

    assert gauge.peak == 5
    assert [result.status for result in results] == [RunStatus.FINISHED] * 5


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
async def test_no_more_runs_than_the_cap_are_active_at_once(monkeypatch, cap):
    use_scripted_customers(monkeypatch, 5)
    gauge = Gauge()

    results = await run(module_of(gauge, after(0.03)), make_scenarios(5), max_parallel=cap)

    assert gauge.peak == cap
    assert [result.status for result in results] == [RunStatus.FINISHED] * 5


async def test_waiting_for_a_free_slot_does_not_count_against_the_timeout(monkeypatch):
    """With one slot the second run waits 0.2 s in line. Its own 0.2 s fits the timeout of 0.35 s only if the wait is not counted."""
    use_scripted_customers(monkeypatch, 2)

    results = await run(module_of(Gauge(), after(0.2)), make_scenarios(2), max_parallel=1, default_timeout=0.35)

    assert [result.status for result in results] == [RunStatus.FINISHED, RunStatus.FINISHED]


@pytest.mark.parametrize("cap", [0, -1])
async def test_a_cap_below_one_is_refused_before_anything_starts(cap):
    module = module_of(Gauge(), after(0))

    with pytest.raises(ValueError, match="must be >= 1"):
        await run(module, make_scenarios(2), max_parallel=cap)

    assert module.opened == 0


# --- failures stay with the run that has them ---

async def test_a_failed_run_and_a_timed_out_run_do_not_stop_the_others(monkeypatch):
    use_scripted_customers(monkeypatch, 5)

    async def act(message):
        if message == "Hi from 1":
            raise TargetError("boom")
        if message == "Hi from 3":
            await asyncio.Event().wait()  # never answers
        return "Hello"

    module = module_of(Gauge(), act)

    results = await run(module, make_scenarios(5, timeouts={3: 0.05}))

    assert [result.status for result in results] == [
        RunStatus.FINISHED,
        RunStatus.FAILED,
        RunStatus.FINISHED,
        RunStatus.TIMED_OUT,
        RunStatus.FINISHED,
    ]
    assert results[1].error == "boom"
    assert results[3].outcome is None
    assert [m.text for m in results[3].transcript.messages] == ["Hi from 3"]  # kept, though it timed out
    assert module.opened == module.closed == 5


# --- each run counts its own cost ---

async def test_each_run_has_its_own_usage_and_the_grand_total_is_their_sum(monkeypatch):
    use_scripted_customers(monkeypatch, 3)
    before = grand_total().input_tokens

    results = await run(module_of(Gauge(), after(0.01)), make_scenarios(3))

    assert [result.usage.calls for result in results] == [3, 3, 3]
    # customer i makes 3 calls of 10*(i+1) input tokens each
    assert [result.usage.input_tokens for result in results] == [30, 60, 90]
    assert len({id(result.usage) for result in results}) == 3
    assert grand_total().input_tokens - before == 30 + 60 + 90


# --- reporting ---

async def test_on_finish_is_called_once_per_run_with_its_result(monkeypatch):
    use_scripted_customers(monkeypatch, 5)
    reported = []

    results = await run(module_of(Gauge(), after(0.01)), make_scenarios(5), on_finish=reported.append)

    assert sorted(result.scenario.name for result in reported) == ["s0", "s1", "s2", "s3", "s4"]
    assert all(isinstance(result, RunResult) for result in reported)
    assert set(map(id, reported)) == set(map(id, results))


async def test_a_callback_that_fails_loses_no_result(monkeypatch, caplog):
    use_scripted_customers(monkeypatch, 3)

    def failing_callback(result):
        raise RuntimeError("printing failed")

    results = await run(module_of(Gauge(), after(0)), make_scenarios(3), on_finish=failing_callback)

    assert [result.status for result in results] == [RunStatus.FINISHED] * 3
    assert "printing failed" in caplog.text


# --- stopping ---

async def test_cancelling_the_batch_stops_every_run_and_closes_every_target(monkeypatch):
    """Ctrl+C must stop all runs, and must not be turned into results."""
    use_scripted_customers(monkeypatch, 3)
    gauge = Gauge()
    module = module_of(gauge, waits_for_others(gauge, 99))  # nobody ever gets an answer
    task = asyncio.create_task(run_in_parallel(
        module, make_scenarios(3), max_parallel=3, default_timeout=30, max_calls=60
    ))
    async with asyncio.timeout(5):
        while gauge.inside < 3:
            await asyncio.sleep(0.01)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert gauge.inside == 0
    assert module.opened == module.closed == 3
