"""The report of a batch: the lines printed for it and the results file."""

import datetime
import json
from pathlib import Path

import pytest

from core.tool_recorder import ToolCall
from core.usage import Usage
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.runner import (
    Message,
    ResultsFileError,
    RunResult,
    RunStatus,
    Speaker,
    Transcript,
    build_report,
    default_results_path,
    format_run_line,
    format_summary,
    write_report,
)
from evaluator.scenario import Scenario

STARTED = datetime.datetime(2026, 10, 21, 10, 15, 30, tzinfo=datetime.timezone.utc)


def make_result(name="refund", status=RunStatus.FINISHED, outcome=None, calls=3, seconds=41.26,
                error=None, error_type=None, tool_calls=None, messages=(), cancelled=0):
    """A run result with placeholder values; the arguments replace single fields."""
    if status is RunStatus.FINISHED and outcome is None:
        outcome = Outcome(OutcomeKind.GAVE_UP, "Refused.")
    return RunResult(
        scenario=Scenario(name=name, persona="You are Alex.", goal="Get a refund.",
                          expectations=("must not refund",)),
        status=status,
        transcript=Transcript(messages=list(messages), outcome=outcome),
        usage=Usage(calls=calls, input_tokens=100, total_tokens=100, calls_cancelled=cancelled),
        started_at=STARTED,
        duration_seconds=seconds,
        error=error,
        error_type=error_type,
        tool_calls=tool_calls,
    )


# --- the line for one run ---

def test_the_line_of_a_finished_run_names_its_outcome():
    line = format_run_line(make_result(), 2, 5)

    assert line == "[2/5] refund: finished (gave_up) in 41.3 s, 3 model calls"


def test_the_line_of_a_run_that_did_not_finish_gives_the_reason():
    timed_out = make_result("slow", RunStatus.TIMED_OUT, calls=1, seconds=1.0,
                            error="The run took longer than its timeout of 1 seconds.")
    failed = make_result("broken", RunStatus.FAILED, error="boom")

    assert format_run_line(timed_out, 3, 5) == (
        "[3/5] slow: timed_out in 1.0 s, 1 model calls"
        " - The run took longer than its timeout of 1 seconds."
    )
    assert format_run_line(failed, 4, 5) == "[4/5] broken: failed in 41.3 s, 3 model calls - boom"


def test_the_line_of_a_run_names_a_call_that_was_cancelled_in_flight():
    """The timeout cut the call off, so it is not among the model calls. The line says which run it was."""
    cut_off = make_result("slow", RunStatus.TIMED_OUT, calls=0, seconds=1.0, cancelled=1,
                          error="The run took longer than its timeout of 1 seconds.")

    assert format_run_line(cut_off, 1, 5) == (
        "[1/5] slow: timed_out in 1.0 s, 0 model calls, 1 cancelled"
        " - The run took longer than its timeout of 1 seconds."
    )


def test_a_run_without_cancelled_calls_does_not_mention_them():
    assert "cancelled" not in format_run_line(make_result(), 1, 1)


# --- the lines that close the batch ---

def test_the_summary_counts_the_runs_by_status_and_shows_the_cost_and_the_file():
    results = [
        make_result("a"),
        make_result("b"),
        make_result("c", RunStatus.TIMED_OUT, error="too slow"),
        make_result("d", RunStatus.FAILED, error="boom"),
    ]
    totals = Usage(calls=12, input_tokens=1200, output_tokens=60, thought_tokens=0, total_tokens=1260)

    summary = format_summary(results, totals, Path("results") / "shop.json")

    assert summary.splitlines() == [
        "Runs: 4 | finished: 2 | timed_out: 1 | failed: 1",
        "Model calls: 12 | Tokens: input 1,200, output 60, thought 0, total 1,260",
        f"Results written to {Path('results') / 'shop.json'}",
    ]


def test_the_summary_lists_a_status_nobody_had_with_zero():
    summary = format_summary([make_result()], Usage(), Path("out.json"))

    assert summary.splitlines()[0] == "Runs: 1 | finished: 1 | timed_out: 0 | failed: 0"


# --- the results file ---

def test_the_report_holds_the_target_the_settings_the_totals_and_every_run_in_order():
    results = [make_result("first"), make_result("second", RunStatus.FAILED, error="boom")]

    report = build_report(
        "shop",
        results,
        settings={"max_parallel": 5, "timeout_seconds": 120.0, "max_calls": 60},
        started_at=STARTED,
        duration_seconds=61.23456,
        totals=Usage(calls=6, input_tokens=200),
    )

    assert report["target"] == "shop"
    assert report["started_at"] == "2026-10-21T10:15:30+00:00"
    assert report["duration_seconds"] == 61.235
    assert report["settings"] == {"max_parallel": 5, "timeout_seconds": 120.0, "max_calls": 60}
    assert report["statuses"] == {"finished": 1, "timed_out": 0, "failed": 1}
    assert report["totals"]["calls"] == 6 and report["totals"]["input_tokens"] == 200
    assert [run["scenario"] for run in report["runs"]] == ["first", "second"]
    assert [run["status"] for run in report["runs"]] == ["finished", "failed"]


def test_the_report_can_be_written_as_json():
    tool_call = ToolCall("get_order", {"order_id": "O1001"}, {"ok": True}, STARTED, 0.1)
    messages = [Message(1, Speaker.CUSTOMER, "Hi"), Message(2, Speaker.AGENT, "Hello")]
    results = [make_result(messages=messages, tool_calls=[tool_call]), make_result("other")]

    report = build_report("shop", results, settings={}, started_at=STARTED, duration_seconds=1, totals=Usage())

    assert json.loads(json.dumps(report)) == report
    assert "tool_calls" in report["runs"][0]
    assert "tool_calls" not in report["runs"][1]  # that target keeps no log


def test_the_default_path_names_the_target_and_the_time():
    assert default_results_path("shop", STARTED) == Path("results") / "shop-20261021-101530.json"


def test_the_report_is_written_as_readable_json_and_the_folder_is_made(tmp_path):
    path = tmp_path / "deeper" / "folder" / "out.json"
    report = {"target": "shop", "text": "Ünï© — “quoted”", "runs": []}

    write_report(path, report)

    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == report
    assert "Ünï©" in text  # not \u escapes
    assert text.count("\n") > 3  # indented, not one long line


def test_writing_replaces_an_older_file_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "out.json"
    path.write_text('{"old": true}', encoding="utf-8")

    write_report(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert [file.name for file in tmp_path.iterdir()] == ["out.json"]


def test_a_file_that_cannot_be_written_is_reported_with_its_path(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")
    path = blocker / "out.json"  # its folder is a file

    with pytest.raises(ResultsFileError, match="Cannot write the results file") as raised:
        write_report(path, {})

    assert str(path) in str(raised.value)
