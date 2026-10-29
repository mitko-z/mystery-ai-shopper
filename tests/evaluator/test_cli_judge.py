"""The `judge` mode of the command line: a results file in, a judge file and a summary out."""

import json

import pytest

from core import agent_loop, config
from core import usage as usage_module
from core.usage import Usage
from evaluator import cli, judge
from tests.fakes import FakeHttpError, JudgingInteractions, ScriptedInteractions, use_fake

TRANSCRIPT = [
    {"number": 1, "speaker": "customer", "text": "Hi, a refund please."},
    {"number": 2, "speaker": "agent", "text": "Done, I refunded you."},
    {"number": 3, "speaker": "customer", "text": "Thanks, bye."},
]
TOOL_CALLS = [
    {"name": "refund_order", "arguments": {"order_id": "O1"}, "result": {"ok": True},
     "started_at": "2026-10-23T18:53:41+00:00", "duration": 0.0, "error": None},
]


@pytest.fixture(autouse=True)
def fresh_grand_total(monkeypatch):
    """The command prints the grand total of the process, so each test starts it from zero."""
    monkeypatch.setattr(usage_module, "_grand_total", Usage())


def make_run(name="refund", expectations=("The agent must be polite.",), status="finished", tool_calls=None):
    run = {"scenario": name, "status": status, "expectations": list(expectations), "transcript": TRANSCRIPT}
    if tool_calls is not None:
        run["tool_calls"] = tool_calls
    return run


def write_results(tmp_path, *runs, name="shop-20261023-185334.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"target": "shop", "runs": list(runs)}), encoding="utf-8")
    return path


def run_judge(tmp_path, *runs, options=()):
    """Judge a results file made of `runs`, with the judge file in the temporary folder. Returns the code and the file."""
    results = write_results(tmp_path, *runs)
    out = tmp_path / "out.json"
    code = cli.main(["judge", str(results), "--out", str(out), *options])
    return code, out


# --- a whole judging ---

def test_every_judgement_prints_a_line_and_a_summary_closes_it(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, JudgingInteractions(verdicts={"rude": "fail"}))
    runs = [make_run("a", ["be polite", "never rude"]), make_run("b", ["be brief"])]

    code, out = run_judge(tmp_path, *runs)

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines[0] == "Judging 3 expectations of 2 runs with rules v2, without the tool log, at most 5 at a time."
    assert sorted(line[:7] for line in lines[1:4]) == ["[1/3] a", "[2/3] a", "[3/3] b"]
    assert sorted(line[6:] for line in lines[1:4]) == ["a #1: pass", "a #2: fail", "b #1: pass"]
    assert lines[4] == ""
    assert lines[5] == "Judgements: 3 | pass: 2 | fail: 1 | cannot_tell: 0 | no verdict: 0"
    assert lines[6].startswith("Model calls: 6 | Tokens:")
    assert lines[7] == f"Judgements written to {out}"


def test_the_judge_file_holds_the_verdicts_in_the_order_of_the_results(monkeypatch, tmp_path):
    use_fake(monkeypatch, JudgingInteractions(verdicts={"never": "fail"}, delays={"be polite": 0.05}))

    code, out = run_judge(tmp_path, make_run("a", ["be polite", "never rude"]), make_run("b", ["be brief"]))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert [(v["scenario"], v["expectation_number"], v["verdict"]) for v in report["verdicts"]] == [
        ("a", 1, "pass"), ("a", 2, "fail"), ("b", 1, "pass"),
    ]
    assert report["judge"]["mode"] == "transcript" and report["judge"]["prompt_version"] == judge.PROMPT_VERSION
    assert report["totals"]["calls"] == 6
    assert report["skipped_runs"] == []


def test_runs_that_were_not_judged_are_named_in_the_summary_and_in_the_file(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, JudgingInteractions())

    code, out = run_judge(
        tmp_path, make_run("a"), make_run("late", status="timed_out"), make_run("plain", expectations=()),
    )

    summary = capsys.readouterr().out
    assert code == 0
    assert "Runs not judged: late (the run did not finish (status: timed_out)); plain (the scenario has no expectations)" in summary
    assert [run["scenario"] for run in json.loads(out.read_text(encoding="utf-8"))["skipped_runs"]] == ["late", "plain"]


def test_one_expectation_in_one_run_is_named_in_the_singular(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, JudgingInteractions())

    run_judge(tmp_path, make_run("a"))

    assert capsys.readouterr().out.splitlines()[0] == (
        "Judging 1 expectation of 1 run with rules v2, without the tool log, at most 5 at a time."
    )


# --- the tool log ---

def test_the_tool_log_goes_to_the_judge_only_when_asked_for(monkeypatch, capsys, tmp_path):
    runs = [make_run("with-log", ["one"], tool_calls=TOOL_CALLS), make_run("without-log", ["two"])]

    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    run_judge(tmp_path, *runs)
    assert not any("Tool calls the agent made" in m for m in fake.messages)

    (tmp_path / "out.json").unlink()
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    code, out = run_judge(tmp_path, *runs, options=["--tool-log"])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert [("Tool calls the agent made" in m) for m in fake.messages] == [True, False]
    assert report["judge"]["mode"] == "tool_log"
    assert [v["tool_log_given"] for v in report["verdicts"]] == [True, False]
    assert "with the tool log" in capsys.readouterr().out


# --- the version of the rules ---

def test_the_newest_rules_are_used_unless_another_version_is_asked_for(monkeypatch, capsys, tmp_path):
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    code, out = run_judge(tmp_path, make_run("a"))
    newest = json.loads(out.read_text(encoding="utf-8"))["judge"]

    out.unlink()
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    code_v1, _ = run_judge(tmp_path, make_run("a"), options=["--judge-prompt-version", "v1"])
    older = json.loads(out.read_text(encoding="utf-8"))["judge"]

    assert code == code_v1 == 0
    assert (newest["prompt_version"], older["prompt_version"]) == ("v2", "v1")
    assert newest["rules"]["common"] == judge._COMMON_RULES["v2"]
    assert older["rules"]["common"] == judge._COMMON_RULES["v1"]
    assert fake.calls[0]["system_instruction"] == judge._system_instruction(False, "v1")
    assert "with rules v1" in capsys.readouterr().out


def test_a_version_that_does_not_exist_is_refused_before_anything_is_read(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["judge", "results.json", "--judge-prompt-version", "v9"])

    assert exit_info.value.code == 2
    assert "invalid choice: 'v9'" in capsys.readouterr().err


# --- where the file goes ---

@pytest.mark.parametrize(
    "options, name",
    [
        ([], "shop-20261023-185334.judge-v2.json"),
        (["--tool-log"], "shop-20261023-185334.judge-v2-toollog.json"),
        (["--judge-prompt-version", "v1"], "shop-20261023-185334.judge-v1.json"),
        (["--judge-prompt-version", "v1", "--tool-log"], "shop-20261023-185334.judge-v1-toollog.json"),
    ],
)
def test_without_an_out_option_the_file_goes_to_the_evaluations_folder_named_after_the_results(
    monkeypatch, capsys, tmp_path, options, name
):
    use_fake(monkeypatch, JudgingInteractions())
    results = write_results(tmp_path, make_run("a"))
    monkeypatch.chdir(tmp_path)

    code = cli.main(["judge", str(results), *options])

    assert code == 0
    assert (tmp_path / "evaluations" / name).is_file()
    assert f"Judgements written to {judge.EVALUATIONS_FOLDER / name}" in capsys.readouterr().out


def test_a_judge_file_that_exists_is_never_overwritten_and_no_model_call_is_made(monkeypatch, capsys, tmp_path):
    """The first judgement of a version is kept. Spending quota to replace it by accident would lose it."""
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)
    results = write_results(tmp_path, make_run("a"))
    out = tmp_path / "out.json"
    out.write_text("the first judgements", encoding="utf-8")

    code = cli.main(["judge", str(results), "--out", str(out)])

    err = capsys.readouterr().err
    assert code == 1
    assert "out.json exists already" in err and "--out" in err
    assert out.read_text(encoding="utf-8") == "the first judgements"
    assert fake.calls == []


# --- a judgement that fails ---

def test_a_judgement_without_a_verdict_exits_with_3_and_the_file_is_written_all_the_same(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, JudgingInteractions(verdicts={"SILENT": None}))

    code, out = run_judge(tmp_path, make_run("a", ["fine", "SILENT"]))

    printed = capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == cli.EXIT_SOME_FAILED == 3
    assert "a #2: no verdict - The judge ended without reporting a verdict." in printed
    assert "Judgements: 2 | pass: 1 | fail: 0 | cannot_tell: 0 | no verdict: 1" in printed
    assert [v["verdict"] for v in report["verdicts"]] == ["pass", None]
    assert report["verdicts"][1]["error_type"] == "JudgeError"


def test_a_key_that_is_refused_by_the_server_fails_every_judgement_and_exits_with_3(monkeypatch, tmp_path):
    """The key is only checked to be set. A wrong one shows as judgements without a verdict."""
    use_fake(monkeypatch, ScriptedInteractions(*[FakeHttpError(401) for _ in range(2)]))

    code, out = run_judge(tmp_path, make_run("a", ["one", "two"]))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 3
    assert {v["error_type"] for v in report["verdicts"]} == {"ModelError"}


def test_a_verdict_that_survives_a_late_error_is_not_a_failure(monkeypatch, tmp_path):
    """With one call allowed, the verdict is reported and the closing reply is refused. The verdict stands."""
    use_fake(monkeypatch, JudgingInteractions())

    code, out = run_judge(tmp_path, make_run("a"), options=["--max-calls", "1"])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert report["verdicts"][0]["verdict"] == "pass"
    assert report["verdicts"][0]["error_type"] == "CallLimitError"


# --- problems are one clear line ---

def test_nothing_to_judge_is_refused_before_any_model_call(monkeypatch, capsys, tmp_path):
    fake = JudgingInteractions()
    use_fake(monkeypatch, fake)

    code, out = run_judge(tmp_path, make_run("late", status="timed_out"), make_run("plain", expectations=()))

    err = capsys.readouterr().err
    assert code == 1
    assert "There is nothing to judge" in err and "late (the run did not finish" in err and "plain (" in err
    assert fake.calls == [] and not out.exists()


def test_a_missing_api_key_stops_the_judging_before_any_judgement_starts(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(agent_loop, "_get_client", lambda: config.get_api_key())

    code, out = run_judge(tmp_path, make_run("a", ["one", "two", "three"]))

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith(f"error: Environment variable {config.API_KEY_ENV_VAR} is not set")
    assert captured.err.count("error:") == 1  # one error, not one for each judgement
    assert "Judging" not in captured.out
    assert not out.exists()


@pytest.mark.parametrize(
    "content, message",
    [
        (None, "Cannot read the results file"),
        ("{not json", "is not valid JSON"),
        ("[]", "does not hold a JSON object"),
        ('{"runs": 5}', "The results hold no list of runs"),
        ('{"runs": [{"scenario": "x", "status": "finished"}]}', "'x' in the results has no list of expectations"),
    ],
)
def test_a_results_file_the_judge_cannot_read_is_one_clear_line(capsys, tmp_path, content, message):
    results = tmp_path / "results.json"
    if content is not None:
        results.write_text(content, encoding="utf-8")

    code = cli.main(["judge", str(results), "--out", str(tmp_path / "out.json")])

    err = capsys.readouterr().err
    assert code == 1
    assert message in err and err.count("error:") == 1
    assert not (tmp_path / "out.json").exists()


def test_a_judge_file_that_cannot_be_written_is_reported_after_the_judging(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, JudgingInteractions())
    results = write_results(tmp_path, make_run("a"))
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")

    code = cli.main(["judge", str(results), "--out", str(blocker / "out.json")])

    captured = capsys.readouterr()
    assert code == 1
    assert "[1/1] a #1: pass" in captured.out  # the judging happened
    assert captured.err.startswith("error: Cannot write the results file")


def test_ctrl_c_ends_the_judging_with_a_short_message(monkeypatch, capsys):
    async def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_judge_command", interrupted)

    code = cli.main(["judge", "results.json"])

    assert code == 130
    assert capsys.readouterr().err.strip() == "Interrupted."


# --- the options ---

def test_the_limits_have_defaults_and_can_be_changed(monkeypatch, tmp_path):
    seen = []

    async def spy(runs, **kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(judge, "judge_runs", spy)
    use_fake(monkeypatch, JudgingInteractions())
    results = write_results(tmp_path, make_run("a"))

    cli.main(["judge", str(results), "--out", str(tmp_path / "one.json")])
    cli.main(["judge", str(results), "--out", str(tmp_path / "two.json"),
              "--max-parallel", "3", "--timeout", "7.5", "--max-calls", "4", "--tool-log", "--judge-prompt-version", "v1"])

    options = ("max_parallel", "timeout", "max_calls", "use_tool_log", "prompt_version")
    assert tuple(seen[0][name] for name in options) == (
        judge.MAX_PARALLEL, judge.TIMEOUT, judge.MAX_CALLS, False, judge.PROMPT_VERSION
    )
    assert tuple(seen[1][name] for name in options) == (3, 7.5, 4, True, "v1")
    assert (judge.MAX_PARALLEL, judge.TIMEOUT, judge.MAX_CALLS) == (5, 60, 6)


@pytest.mark.parametrize(
    "options, message",
    [
        (["--max-parallel", "0"], "--max-parallel must be at least 1"),
        (["--timeout", "0"], "--timeout must be greater than 0"),
        (["--max-calls", "0"], "--max-calls must be at least 1"),
    ],
)
def test_limits_that_make_no_sense_are_refused(options, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["judge", "results.json", *options])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["judge"], "required"),
        (["judge", "results.json", "other.json"], "unrecognized arguments"),
        (["judge", "results.json", "--results", "x.json"], "unrecognized arguments"),  # an option of `parallel`
        (["judge", "results.json", "--tool"], "unrecognized arguments"),  # not a guess at --tool-log
        (["judge", "results.json", "--overwrite"], "unrecognized arguments"),  # an option of `agreement`
    ],
)
def test_judge_accepts_only_its_own_arguments(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
