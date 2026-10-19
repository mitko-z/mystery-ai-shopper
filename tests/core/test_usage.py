import asyncio

import pytest

from core import usage as usage_module
from core.usage import Usage, grand_total, record, track_usage
from tests.fakes import reported_usage


@pytest.fixture(autouse=True)
def fresh_grand_total(monkeypatch):
    """Give each test its own grand total, so tests do not see each other's numbers."""
    monkeypatch.setattr(usage_module, "_grand_total", Usage())


def test_a_new_usage_is_all_zero():
    assert Usage() == Usage(0, 0, 0, 0, 0, 0)


def test_add_counts_the_call_and_its_tokens():
    usage = Usage()

    usage.add(reported_usage(input=100, output=20, thought=30, total=150))

    assert usage == Usage(
        calls=1, input_tokens=100, output_tokens=20, thought_tokens=30, total_tokens=150
    )


def test_add_sums_over_calls():
    usage = Usage()

    usage.add(reported_usage(input=100, output=20, thought=30, total=150))
    usage.add(reported_usage(input=200, output=5, thought=0, total=205))

    assert usage == Usage(
        calls=2, input_tokens=300, output_tokens=25, thought_tokens=30, total_tokens=355
    )


def test_a_call_without_a_report_is_counted_and_flagged():
    usage = Usage()

    usage.add(None)

    assert usage == Usage(calls=1, calls_without_usage=1)


def test_a_missing_count_in_a_report_is_zero():
    usage = Usage()

    usage.add(reported_usage(input=100, total=100))

    assert usage == Usage(calls=1, input_tokens=100, total_tokens=100)
    assert usage.calls_without_usage == 0


def test_record_in_a_tracked_run_counts_into_the_run_and_the_grand_total():
    with track_usage() as run:
        record(reported_usage(input=10, total=10))

    assert run == Usage(calls=1, input_tokens=10, total_tokens=10)
    assert grand_total() == run


def test_record_outside_a_run_counts_only_into_the_grand_total():
    record(reported_usage(input=10, total=10))

    assert grand_total() == Usage(calls=1, input_tokens=10, total_tokens=10)


def test_a_finished_run_no_longer_counts():
    with track_usage() as run:
        record(reported_usage(input=10, total=10))

    record(reported_usage(input=5, total=5))

    assert run.calls == 1
    assert run.total_tokens == 10
    assert grand_total().calls == 2


def test_a_run_that_fails_stops_counting_and_keeps_what_it_counted():
    with pytest.raises(RuntimeError):
        with track_usage() as run:
            record(reported_usage(input=10, total=10))
            raise RuntimeError("the run failed")

    record(reported_usage(input=5, total=5))

    assert run == Usage(calls=1, input_tokens=10, total_tokens=10)
    assert grand_total().calls == 2


@pytest.mark.anyio
async def test_concurrent_runs_are_counted_separately():
    async def run(calls, tokens):
        with track_usage() as usage:
            for _ in range(calls):
                record(reported_usage(input=tokens, total=tokens))
                await asyncio.sleep(0)  # let the other run count in between
        return usage

    first, second = await asyncio.gather(run(3, 10), run(5, 100))

    assert first == Usage(calls=3, input_tokens=30, total_tokens=30)
    assert second == Usage(calls=5, input_tokens=500, total_tokens=500)
    assert grand_total() == Usage(calls=8, input_tokens=530, total_tokens=530)


@pytest.mark.anyio
async def test_a_task_started_inside_a_run_counts_into_that_run():
    async def child():
        record(reported_usage(input=10, total=10))

    with track_usage() as run:
        await asyncio.create_task(child())

    assert run == Usage(calls=1, input_tokens=10, total_tokens=10)
