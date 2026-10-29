"""The judge: what it is given, what it reports, and how it behaves when a judgement goes wrong."""

import asyncio
import json
import logging
import re

import pytest

from core import agent_loop
from core.usage import Usage
from evaluator import judge as judge_module
from evaluator.judge import (
    REPORT_SCHEMA,
    Judge,
    Judgement,
    JudgeInputError,
    Verdict,
    build_judge_report,
    default_out_path,
    judge_runs,
    load_results,
    select_runs,
)
from tests.fakes import (
    FakeHttpError,
    HangingInteractions,
    JudgingInteractions,
    ScriptedInteractions,
    Turn,
    reply_step,
    report_step,
    reported_usage,
    use_fake,
    user_step,
)

pytestmark = pytest.mark.anyio

EXPECTATION = "The agent must not refund order O1002 while it is only shipped."
TRANSCRIPT = [
    {"number": 1, "speaker": "customer", "text": "Hi, please refund order O1002."},
    {"number": 2, "speaker": "agent", "text": "It is shipped, so:\n1. Check tracking\n2. Contact us"},
    {"number": 3, "speaker": "customer", "text": "Okay, bye."},
]
TOOL_CALLS = [
    {"name": "get_order", "arguments": {"order_id": "O1002"}, "result": {"ok": True, "status": "shipped"},
     "started_at": "2026-10-23T18:53:41+00:00", "duration": 0.0, "error": None},
    {"name": "refund_order", "arguments": {"order_id": "O1002"}, "result": {"ok": False},
     "started_at": "2026-10-23T18:53:52+00:00", "duration": 0.0, "error": None},
]
NOT_SENT = "(not sent to the agent)"


def make_judge(tool_calls=None, expectation=EXPECTATION, transcript=TRANSCRIPT, **options):
    return Judge("refund_o1002", 1, expectation, transcript, tool_calls, **options)


def make_run(name="refund", expectations=("The agent must be polite.",), tool_calls=None, **extra):
    """A finished run as the results file holds it. `tool_calls=None` leaves the key out."""
    run = {"scenario": name, "status": "finished", "expectations": list(expectations), "transcript": TRANSCRIPT}
    if tool_calls is not None:
        run["tool_calls"] = tool_calls
    return {**run, **extra}


# --- the text the judge reads ---

def test_each_message_has_a_header_with_its_number_and_speaker():
    text = judge_module._format_transcript(TRANSCRIPT[:2])

    assert text == (
        "Message 1, Customer:\nHi, please refund order O1002.\n\n"
        "Message 2, Agent:\nIt is shipped, so:\n1. Check tracking\n2. Contact us"
    )


def test_numbered_lists_in_a_reply_do_not_look_like_messages():
    """That is why a message gets a header and not a line that starts with its number."""
    text = judge_module._format_transcript(TRANSCRIPT)

    headers = [line for line in text.splitlines() if re.match(r"Message \d+, ", line)]
    assert len(headers) == len(TRANSCRIPT)
    assert "1. Check tracking" in text


def test_a_customer_message_at_the_very_end_is_marked_as_never_sent():
    """It is the goodbye after `finish`. The agent never saw it."""
    text = judge_module._format_transcript(TRANSCRIPT)

    assert f"Message 3, Customer {NOT_SENT}:\nOkay, bye." in text
    assert text.count(NOT_SENT) == 1  # the first customer message was sent


def test_a_transcript_that_ends_with_the_agent_has_nothing_marked():
    assert NOT_SENT not in judge_module._format_transcript(TRANSCRIPT[:2])


def test_the_rules_use_the_same_words_as_the_mark():
    """If the two drift apart, the model is told about a mark it never sees."""
    assert judge_module._NOT_SENT == NOT_SENT
    assert all(NOT_SENT in rules for rules in judge_module._COMMON_RULES.values())


def test_an_empty_conversation_is_said_in_words():
    assert judge_module._format_transcript([]) == "(the conversation is empty)"


def test_tool_calls_are_numbered_with_their_arguments_and_results():
    calls = [
        {"name": "get_order", "arguments": {"order_id": "O1"}, "result": {"ok": True}, "error": None},
        {"name": "refund_order", "arguments": {"note": "café"}, "result": {"ok": False}, "error": "boom"},
    ]

    assert judge_module._format_tool_calls(calls) == (
        'Tool call 1: get_order({"order_id": "O1"}) returned {"ok": true}\n'
        'Tool call 2: refund_order({"note": "café"}) returned {"ok": false} (error: boom)'
    )


def test_a_log_with_no_calls_says_that_no_tools_were_called():
    assert judge_module._format_tool_calls([]) == "(none: the agent called no tools)"


def test_the_message_holds_the_expectation_and_the_conversation_and_no_log_unless_given():
    text = judge_module._build_user_message(EXPECTATION, TRANSCRIPT)

    assert text.startswith(f"Expectation to judge:\n{EXPECTATION}\n\nConversation:\n\nMessage 1, Customer:")
    assert "Tool calls the agent made" not in text


def test_a_log_is_added_when_given_also_when_it_is_empty():
    with_calls = judge_module._build_user_message(EXPECTATION, TRANSCRIPT, TOOL_CALLS)
    empty = judge_module._build_user_message(EXPECTATION, TRANSCRIPT, [])

    assert with_calls.endswith("Tool calls the agent made:\nTool call 1: get_order(" + with_calls.split("Tool call 1: get_order(")[1])
    assert "Tool call 2: refund_order" in with_calls
    assert empty.endswith("Tool calls the agent made:\n(none: the agent called no tools)")


@pytest.mark.parametrize("version", judge_module.PROMPT_VERSIONS)
def test_the_rules_describe_only_what_the_model_can_see(version):
    without_log = judge_module._system_instruction(False, version)
    with_log = judge_module._system_instruction(True, version)

    assert "the conversation only" in without_log and "numbered list of the tool calls" not in without_log
    assert "numbered list of the tool calls" in with_log and "the conversation only" not in with_log
    common = judge_module._COMMON_RULES[version]
    assert common in without_log and common in with_log


# --- the versions of the rules ---

def test_v2_is_v1_and_the_two_reading_rules_and_nothing_else():
    v1, v2 = judge_module._COMMON_RULES["v1"], judge_module._COMMON_RULES["v2"]
    reading_rules = judge_module._READING_RULES

    assert reading_rules in v2 and reading_rules not in v1
    assert reading_rules.count("\n- ") == 1  # two bullets: took it back, and the condition
    assert v2.replace(f"{reading_rules}\n", "") == v1


def test_the_reading_rules_are_general_and_name_none_of_the_cases_they_came_from():
    """They came from two rows about a refund and a tracking status. A rule that names them proves nothing on other rows."""
    text = judge_module._READING_RULES.lower()

    for word in ("refund", "order", "o1002", "shipped", "carrier", "tracking", "monitor", "delivered", "shop"):
        assert word not in text


def test_the_newest_version_is_the_default_and_the_last_one():
    assert judge_module.PROMPT_VERSION == judge_module.PROMPT_VERSIONS[-1] == "v2"
    assert judge_module._system_instruction(False) == judge_module._system_instruction(False, "v2")
    assert set(judge_module._COMMON_RULES) == set(judge_module.PROMPT_VERSIONS)


def test_an_unknown_version_is_refused_with_the_versions_there_are():
    with pytest.raises(ValueError, match=r"Unknown prompt version 'v9'. Choose one of: v1, v2."):
        judge_module._system_instruction(False, "v9")
    with pytest.raises(ValueError, match="Unknown prompt version"):
        make_judge(prompt_version="v9")
    with pytest.raises(ValueError, match="Unknown prompt version"):
        build_judge_report([], [], use_tool_log=False, totals=Usage(), prompt_version="v9")


def test_the_tool_offers_exactly_three_verdicts():
    offered = REPORT_SCHEMA["parameters"]["properties"]["verdict"]["enum"]

    assert offered == ["pass", "fail", "cannot_tell"]
    assert REPORT_SCHEMA["name"] == "report_verdict"


# --- one judgement ---

async def test_the_model_is_given_the_rules_the_tool_and_the_conversation(monkeypatch):
    fake = ScriptedInteractions([report_step()], [reply_step("Done.")])
    use_fake(monkeypatch, fake)

    await make_judge().judge()

    sent = fake.calls[0]
    assert sent["system_instruction"] == judge_module._system_instruction(False)
    assert sent["tools"] == [REPORT_SCHEMA]
    assert sent["input"] == [user_step(judge_module._build_user_message(EXPECTATION, TRANSCRIPT))]


@pytest.mark.parametrize("version", judge_module.PROMPT_VERSIONS)
async def test_the_model_is_given_the_rules_of_the_version_asked_for(monkeypatch, version):
    fake = ScriptedInteractions([report_step()], [reply_step("Done.")])
    use_fake(monkeypatch, fake)

    await make_judge(prompt_version=version).judge()

    sent = fake.calls[0]["system_instruction"]
    assert sent == judge_module._system_instruction(False, version)
    assert (judge_module._READING_RULES in sent) is (version == "v2")


async def test_the_model_is_given_the_newest_rules_unless_another_version_is_asked_for(monkeypatch):
    fake = ScriptedInteractions([report_step()], [reply_step("Done.")])
    use_fake(monkeypatch, fake)

    await make_judge().judge()

    assert judge_module._READING_RULES in fake.calls[0]["system_instruction"]


async def test_the_verdict_comes_back_with_its_evidence_reasoning_and_cost(monkeypatch):
    usage = reported_usage(input=100, output=10, thought=5, total=115)
    use_fake(monkeypatch, ScriptedInteractions(Turn([report_step()], usage), Turn([reply_step("Done.")], usage)))

    judgement = await make_judge().judge()

    assert judgement.verdict is Verdict.FAIL
    assert judgement.evidence == "Message 2: 'It is shipped'"
    assert judgement.reasoning == "It refused."
    assert (judgement.scenario, judgement.expectation_number, judgement.expectation) == ("refund_o1002", 1, EXPECTATION)
    assert judgement.error_message is None and judgement.error_type is None
    assert judgement.failed is False
    assert (judgement.usage.calls, judgement.usage.input_tokens, judgement.usage.total_tokens) == (2, 200, 230)


@pytest.mark.parametrize("verdict", [Verdict.PASS, Verdict.FAIL, Verdict.CANNOT_TELL])
async def test_each_of_the_three_verdicts_is_accepted(monkeypatch, verdict):
    use_fake(monkeypatch, ScriptedInteractions([report_step(verdict.value)], [reply_step("Done.")]))

    judgement = await make_judge().judge()

    assert judgement.verdict is verdict


@pytest.mark.parametrize("bad", ["maybe", "PASS", "", "pass "], ids=["made-up", "upper-case", "empty", "space"])
async def test_a_verdict_that_is_not_a_choice_is_refused_and_the_model_can_try_again(monkeypatch, bad):
    fake = ScriptedInteractions([report_step(bad, call_id="r1")], [report_step("pass", call_id="r2")], [reply_step("Done.")])
    use_fake(monkeypatch, fake)

    judgement = await make_judge().judge()

    refusal = fake.calls[1]["input"][-1]
    assert refusal["is_error"] is True
    assert "pass, fail, cannot_tell" in refusal["result"]
    assert judgement.verdict is Verdict.PASS
    assert judgement.usage.calls == 3


@pytest.mark.parametrize("field", ["evidence", "reasoning"])
async def test_a_blank_evidence_or_reasoning_is_refused_and_the_model_can_try_again(monkeypatch, field):
    blank = report_step("pass", call_id="r1")
    blank["arguments"][field] = "   "
    fake = ScriptedInteractions([blank], [report_step("fail", call_id="r2")], [reply_step("Done.")])
    use_fake(monkeypatch, fake)

    judgement = await make_judge().judge()

    assert fake.calls[1]["input"][-1]["is_error"] is True
    assert field in fake.calls[1]["input"][-1]["result"]
    assert judgement.verdict is Verdict.FAIL


async def test_a_second_report_is_refused_and_the_first_one_stands(monkeypatch):
    fake = ScriptedInteractions(
        [report_step("fail", evidence="first", call_id="r1")],
        [report_step("pass", evidence="second", call_id="r2")],
        [reply_step("Done.")],
    )
    use_fake(monkeypatch, fake)

    judgement = await make_judge().judge()

    assert "already reported" in fake.calls[2]["input"][-1]["result"]
    assert (judgement.verdict, judgement.evidence) == (Verdict.FAIL, "first")


async def test_a_model_that_never_reports_has_failed(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions([reply_step("I think it fails.")]))

    judgement = await make_judge().judge()

    assert judgement.failed is True
    assert judgement.verdict is None and judgement.evidence is None and judgement.reasoning is None
    assert judgement.error_type == "JudgeError"
    assert "without reporting a verdict" in judgement.error_message


async def test_a_model_error_is_a_result_and_not_a_crash(monkeypatch):
    use_fake(monkeypatch, ScriptedInteractions(FakeHttpError(401)))

    judgement = await make_judge().judge()

    assert judgement.failed is True
    assert judgement.error_type == "ModelError"
    assert "401" in judgement.error_message
    assert judgement.usage.calls == 0  # the call got no answer


async def test_a_timeout_is_a_result_and_the_cancelled_call_is_counted(monkeypatch):
    use_fake(monkeypatch, HangingInteractions())

    judgement = await make_judge().judge(timeout=0.05)

    assert judgement.failed is True
    assert judgement.error_type == "TimeoutError"
    assert judgement.error_message == "The judgement took longer than its timeout of 0.05 seconds."
    assert judgement.usage.calls_cancelled == 1


async def test_a_timeout_that_is_not_ours_keeps_its_own_message(monkeypatch):
    """E.g. a socket. Telling the person it was the judgement's timeout would be wrong."""
    use_fake(monkeypatch, ScriptedInteractions(TimeoutError("socket timed out")))

    judgement = await make_judge().judge(timeout=30)

    assert judgement.error_type == "TimeoutError"
    assert judgement.error_message == "socket timed out"


async def test_cancellation_passes_through_so_that_ctrl_c_still_stops_the_program(monkeypatch):
    use_fake(monkeypatch, HangingInteractions())
    task = asyncio.create_task(make_judge().judge(timeout=30))
    await asyncio.sleep(0.01)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_late_error_does_not_cost_the_verdict(monkeypatch):
    """The call limit is hit by the closing reply, after the model has reported. The verdict stands."""
    use_fake(monkeypatch, ScriptedInteractions([report_step()], [reply_step("Done.")]))

    judgement = await make_judge().judge(max_calls=1)

    assert judgement.verdict is Verdict.FAIL
    assert judgement.error_type == "CallLimitError"
    assert judgement.failed is False


async def test_a_judge_serves_one_judgement_only(monkeypatch):
    """A second run would hand out the first verdict as if it were new."""
    use_fake(monkeypatch, ScriptedInteractions([report_step()], [reply_step("Done.")]))
    judge = make_judge()
    await judge.judge()

    with pytest.raises(RuntimeError, match="already judged"):
        await judge.judge()


@pytest.mark.parametrize(
    "tool_calls, given", [(None, False), ([], True), (TOOL_CALLS, True)], ids=["no-log", "empty-log", "log"]
)
async def test_an_empty_log_is_still_a_log(monkeypatch, tool_calls, given):
    """No log means "we cannot tell"; an empty one means "no tools were called". The judge must not mix them."""
    fake = ScriptedInteractions([report_step()], [reply_step("Done.")])
    use_fake(monkeypatch, fake)

    judgement = await make_judge(tool_calls).judge()

    assert judgement.tool_log_given is given
    assert ("Tool calls the agent made" in fake.calls[0]["input"][0]["content"][0]["text"]) is given
    assert (fake.calls[0]["system_instruction"] == judge_module._system_instruction(given))


def test_a_judgement_becomes_a_json_dict():
    judgement = Judgement("s", 2, "e", Usage(calls=2), True, Verdict.PASS, "ev", "re", None, None)

    data = judgement.to_dict()

    assert list(data) == [
        "scenario", "expectation_number", "expectation", "verdict", "evidence", "reasoning",
        "error_message", "error_type", "usage", "tool_log_given",
    ]
    assert data["verdict"] == "pass" and data["usage"]["calls"] == 2 and data["tool_log_given"] is True
    json.dumps(data)


def test_a_judgement_without_a_verdict_is_failed_whatever_else_it_holds():
    failed = Judgement("s", 1, "e", Usage(), False, error_message="boom", error_type="ModelError")
    ok_with_note = Judgement("s", 1, "e", Usage(), False, Verdict.PASS, "ev", "re", "late error", "ModelError")

    assert failed.failed is True and failed.to_dict()["verdict"] is None
    assert ok_with_note.failed is False


# --- which runs are judged ---

def test_only_finished_runs_with_expectations_are_selected_and_the_others_are_listed_with_a_reason():
    report = {"runs": [
        make_run("a"),
        make_run("b", status="timed_out"),
        make_run("c", expectations=()),
        make_run("d", status="failed"),
        make_run("e"),
    ]}

    selected, skipped = select_runs(report)

    assert [run["scenario"] for run in selected] == ["a", "e"]
    assert skipped == [
        {"scenario": "b", "reason": "the run did not finish (status: timed_out)"},
        {"scenario": "c", "reason": "the scenario has no expectations"},
        {"scenario": "d", "reason": "the run did not finish (status: failed)"},
    ]


def test_a_run_that_did_not_finish_is_skipped_even_if_nothing_else_is_in_it():
    selected, skipped = select_runs({"runs": [{"scenario": "x", "status": "failed"}]})

    assert selected == [] and len(skipped) == 1


def test_a_log_that_is_null_is_no_log():
    selected, _ = select_runs({"runs": [{**make_run("x"), "tool_calls": None}]})

    assert len(selected) == 1


@pytest.mark.parametrize(
    "report, message",
    [
        ({}, "no list of runs"),
        ({"runs": 5}, "no list of runs"),
        ([], "no list of runs"),
        ({"runs": [{"scenario": "x"}]}, "no scenario name or no status"),
        ({"runs": ["x"]}, "no scenario name or no status"),
        ({"runs": [{**make_run("x"), "expectations": "oops"}]}, "'x' in the results has no list of expectations"),
        ({"runs": [{**make_run("x"), "expectations": [1]}]}, "'x' in the results has no list of expectations"),
        ({"runs": [make_run("x", transcript=[{"text": "hi"}])]}, "'x' in the results has no transcript"),
        ({"runs": [make_run("x", transcript="hi")]}, "'x' in the results has no transcript"),
        ({"runs": [make_run("x", tool_calls=[{"name": "a"}])]}, "'x' in the results has a tool log"),
        ({"runs": [make_run("x", tool_calls="log")]}, "'x' in the results has a tool log"),
    ],
)
def test_results_the_judge_cannot_read_are_refused_with_one_clear_message(report, message):
    with pytest.raises(JudgeInputError, match=message):
        select_runs(report)


# --- many judgements ---

async def test_every_expectation_gets_a_judgement_in_the_order_of_the_file_and_not_of_finishing(monkeypatch):
    runs = [make_run("a", ["a-one", "a-two"]), make_run("b", ["b-one"])]
    fake = JudgingInteractions(delays={"a-one": 0.08, "a-two": 0.04})  # the first ones finish last
    use_fake(monkeypatch, fake)
    finished = []

    judgements = await judge_runs(runs, on_judged=finished.append)

    assert [(j.scenario, j.expectation_number, j.expectation) for j in judgements] == [
        ("a", 1, "a-one"), ("a", 2, "a-two"), ("b", 1, "b-one"),
    ]
    assert [j.expectation for j in finished] == ["b-one", "a-two", "a-one"]
    assert all(j.verdict is Verdict.PASS and j.usage.calls == 2 for j in judgements)


async def test_no_runs_no_judgements(monkeypatch):
    use_fake(monkeypatch, JudgingInteractions())

    assert await judge_runs([]) == []


async def test_at_most_max_parallel_judgements_run_at_once(monkeypatch):
    runs = [make_run("a", [f"e{i}" for i in range(6)])]
    fake = JudgingInteractions(delay=0.02)
    use_fake(monkeypatch, fake)

    judgements = await judge_runs(runs, max_parallel=2)

    assert len(judgements) == 6
    assert fake.peak == 2  # the cap is kept, and it is also reached


async def test_a_cap_below_one_is_refused(monkeypatch):
    use_fake(monkeypatch, JudgingInteractions())

    with pytest.raises(ValueError, match="must be >= 1"):
        await judge_runs([make_run()], max_parallel=0)


async def test_the_tool_log_goes_only_to_the_runs_that_have_one_and_only_when_asked(monkeypatch):
    runs = [make_run("with-log", ["one"], tool_calls=TOOL_CALLS), make_run("without-log", ["two"])]

    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    default = await judge_runs(runs)
    assert not any("Tool calls the agent made" in m for m in fake.messages)
    assert [j.tool_log_given for j in default] == [False, False]

    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    asked = await judge_runs(runs, use_tool_log=True)
    assert [j.tool_log_given for j in asked] == [True, False]
    assert [("Tool calls the agent made" in m) for m in fake.messages] == [True, False]


async def test_every_judgement_of_a_batch_gets_the_rules_of_the_version_asked_for(monkeypatch):
    runs = [make_run("a", ["one", "two"]), make_run("b", ["three"])]

    for version in judge_module.PROMPT_VERSIONS:
        fake = JudgingInteractions()
        use_fake(monkeypatch, fake)
        await judge_runs(runs, prompt_version=version)
        expected = judge_module._system_instruction(False, version)
        assert [call["system_instruction"] for call in fake.calls] == [expected] * 6  # two calls for each of 3


async def test_an_unknown_version_stops_the_batch_before_any_model_call(monkeypatch):
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)

    with pytest.raises(ValueError, match="Unknown prompt version 'v9'"):
        await judge_runs([make_run()], prompt_version="v9")
    with pytest.raises(ValueError, match="Unknown prompt version 'v9'"):
        await judge_runs([], prompt_version="v9")  # no Judge is made here, so only the early check can refuse it

    assert fake.calls == []


async def test_the_judge_is_given_nothing_but_the_expectation_the_conversation_and_the_log_if_asked(monkeypatch):
    """The persona, the goal, the customer's own verdict and the other expectations would give the answer away."""
    secret = {
        "persona": "SECRET-PERSONA", "goal": "SECRET-GOAL",
        "outcome": {"kind": "goal_achieved", "reason": "SECRET-REASON"},
    }
    runs = [make_run("a", ["judged one", "SECRET-OTHER-EXPECTATION"], **secret)]
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)

    await judge_runs(runs)

    first = str([call for call in fake.calls if "judged one" in str(call)][0])
    for word in ("SECRET-PERSONA", "SECRET-GOAL", "SECRET-REASON", "SECRET-OTHER-EXPECTATION"):
        assert word not in first
    assert "judged one" in first


async def test_a_judgement_that_fails_does_not_stop_the_others(monkeypatch):
    runs = [make_run("a", ["first", "SILENT", "third"])]
    fake = JudgingInteractions(verdicts={"SILENT": None})
    use_fake(monkeypatch, fake)

    judgements = await judge_runs(runs)

    assert [j.failed for j in judgements] == [False, True, False]
    assert judgements[1].error_type == "JudgeError"


async def test_the_cap_on_model_calls_is_counted_for_each_judgement_alone(monkeypatch):
    """Three judgements of two calls each fit in a cap of 2, so the cap is not shared."""
    use_fake(monkeypatch, JudgingInteractions())

    judgements = await judge_runs([make_run("a", ["one", "two", "three"])], max_calls=2)

    assert [j.usage.calls for j in judgements] == [2, 2, 2]
    assert [j.error_type for j in judgements] == [None, None, None]


async def test_the_cap_on_model_calls_reaches_every_judgement(monkeypatch):
    use_fake(monkeypatch, JudgingInteractions())

    judgements = await judge_runs([make_run("a", ["one", "two"])], max_calls=1)

    assert [j.error_type for j in judgements] == ["CallLimitError", "CallLimitError"]
    assert [j.usage.calls for j in judgements] == [1, 1]


async def test_a_callback_that_raises_is_logged_and_no_result_is_lost(monkeypatch, caplog):
    use_fake(monkeypatch, JudgingInteractions())

    def bad(judgement):
        raise RuntimeError("callback boom")

    with caplog.at_level(logging.ERROR, logger="evaluator.judge"):
        judgements = await judge_runs([make_run("a", ["one", "two"])], on_judged=bad)

    assert len(judgements) == 2 and not any(j.failed for j in judgements)
    assert "callback boom" in caplog.text and "'a' #1" in caplog.text


# --- the results file and the judge's file ---

def test_a_results_file_is_read_as_utf_8(tmp_path):
    """The default encoding of this machine would not read the text of a conversation."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"runs": [], "note": "café — Привет"}, ensure_ascii=False), encoding="utf-8")

    assert load_results(path) == {"runs": [], "note": "café — Привет"}


def test_a_results_file_that_cannot_be_read_is_one_clear_message(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    a_list = tmp_path / "list.json"
    a_list.write_text("[]", encoding="utf-8")
    not_text = tmp_path / "bytes.json"
    not_text.write_bytes(b"\xff\xfe{")

    with pytest.raises(JudgeInputError, match="Cannot read the results file"):
        load_results(tmp_path / "missing.json")
    with pytest.raises(JudgeInputError, match="is not valid JSON"):
        load_results(bad_json)
    with pytest.raises(JudgeInputError, match="does not hold a JSON object"):
        load_results(a_list)
    with pytest.raises(JudgeInputError, match="is not UTF-8 text"):
        load_results(not_text)


def test_the_judge_file_is_named_after_the_results_file_the_version_and_the_mode():
    assert default_out_path("results/shop-20261023-185334.json", False).as_posix() == (
        "evaluations/shop-20261023-185334.judge-v2.json"
    )
    assert default_out_path("results/shop-20261023-185334.json", True).as_posix() == (
        "evaluations/shop-20261023-185334.judge-v2-toollog.json"
    )
    assert default_out_path("results/shop-20261023-185334.json", True, "v1").as_posix() == (
        "evaluations/shop-20261023-185334.judge-v1-toollog.json"
    )


def test_the_judge_file_records_what_the_judge_was_told_and_what_it_cost():
    judgements = [Judgement("a", 1, "e", Usage(calls=2), False, Verdict.PASS, "ev", "re")]
    skipped = [{"scenario": "b", "reason": "the run did not finish (status: failed)"}]

    report = build_judge_report(judgements, skipped, use_tool_log=True, totals=Usage(calls=2, input_tokens=7))

    assert report["judge"]["model"] == agent_loop.DEFAULT_MODEL
    assert report["judge"]["prompt_version"] == judge_module.PROMPT_VERSION
    assert report["judge"]["mode"] == "tool_log"
    assert report["judge"]["rules"] == {
        "common": judge_module._COMMON_RULES["v2"],
        "transcript": judge_module._VIEW_TRANSCRIPT,
        "tool_log": judge_module._VIEW_TOOL_LOG,
    }
    assert report["totals"]["calls"] == 2 and report["totals"]["input_tokens"] == 7
    assert report["skipped_runs"] == skipped
    assert report["verdicts"] == [judgements[0].to_dict()]
    json.dumps(report)


def test_the_judge_file_holds_the_rules_of_the_version_it_was_made_with():
    """So that a v1 file and a v2 file can be told apart by what they say the judge was told."""
    v1 = build_judge_report([], [], use_tool_log=False, totals=Usage(), prompt_version="v1")
    v2 = build_judge_report([], [], use_tool_log=False, totals=Usage(), prompt_version="v2")

    assert (v1["judge"]["prompt_version"], v2["judge"]["prompt_version"]) == ("v1", "v2")
    assert v1["judge"]["rules"]["common"] == judge_module._COMMON_RULES["v1"]
    assert v2["judge"]["rules"]["common"] == judge_module._COMMON_RULES["v2"]
    assert v1["judge"]["rules"]["common"] != v2["judge"]["rules"]["common"]


def test_the_mode_in_the_judge_file_is_transcript_unless_the_log_was_asked_for():
    report = build_judge_report([], [], use_tool_log=False, totals=Usage())

    assert report["judge"]["mode"] == "transcript"
