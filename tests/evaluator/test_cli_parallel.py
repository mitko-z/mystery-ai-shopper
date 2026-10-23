"""The `parallel` mode of the command line, and the choice between its modes."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

from core import agent_loop, config
from core import usage as usage_module
from core.usage import Usage
from evaluator import runner
from evaluator.target import Target, TargetError
from tests.evaluator.helpers import make_scenarios, use_scripted_customers
from tests.fakes import FakeHttpError, ScriptedInteractions, use_fake


@pytest.fixture(autouse=True)
def fresh_grand_total(monkeypatch):
    """The batch prints the grand total of the process, so each test starts it from zero."""
    monkeypatch.setattr(usage_module, "_grand_total", Usage())


class AnswerTarget(Target):
    """A target that answers by doing `act`."""

    def __init__(self, act):
        self._act = act

    async def send(self, message):
        return await self._act(message)


async def hello(message):
    return "Hello"


def register_target(monkeypatch, scenarios=(), act=hello):
    """Make a target module `targets.fake` that opens a new AnswerTarget for every run."""
    module = ModuleType("targets.fake")
    module.opened = 0

    @asynccontextmanager
    async def open_target():
        module.opened += 1
        yield AnswerTarget(act)

    module.open_target = open_target
    module.SCENARIOS = {scenario.name: scenario for scenario in scenarios}
    monkeypatch.setitem(sys.modules, "targets.fake", module)
    return module


def run_parallel(tmp_path, *options, target="fake"):
    """Run `parallel` on `target` with the results in the temporary folder. Returns the exit code and the file."""
    results = tmp_path / "out.json"
    code = runner.main(["parallel", target, "--results", str(results), *options])
    return code, results


# --- a whole batch ---

def test_a_batch_runs_every_scenario_and_prints_a_line_each_and_a_summary(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 3)
    register_target(monkeypatch, make_scenarios(3))

    code, results = run_parallel(tmp_path)

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines[0] == "Running 3 scenarios on fake, at most 5 at a time."
    run_lines = lines[1:4]
    assert sorted(line[:5] for line in run_lines) == ["[1/3]", "[2/3]", "[3/3]"]
    assert all("finished (gave_up)" in line for line in run_lines)
    assert sorted(line.split(":")[0][6:] for line in run_lines) == ["s0", "s1", "s2"]
    assert lines[4] == ""
    assert lines[5] == "Runs: 3 | finished: 3 | timed_out: 0 | failed: 0"
    assert lines[6] == "Model calls: 9 | Tokens: input 180, output 0, thought 0, total 180"
    assert lines[7] == f"Results written to {results}"


def test_one_scenario_is_named_in_the_singular(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 1)
    register_target(monkeypatch, make_scenarios(1))

    run_parallel(tmp_path)

    assert capsys.readouterr().out.splitlines()[0] == "Running 1 scenario on fake, at most 5 at a time."


def test_the_results_file_holds_every_run_in_scenario_order(monkeypatch, tmp_path):
    use_scripted_customers(monkeypatch, 3)
    register_target(monkeypatch, make_scenarios(3))

    _, results = run_parallel(tmp_path, "--max-parallel", "2", "--timeout", "30", "--max-calls", "40")

    report = json.loads(results.read_text(encoding="utf-8"))
    assert report["target"] == "fake"
    assert report["settings"] == {"max_parallel": 2, "default_timeout_seconds": 30.0, "max_calls": 40}
    assert report["statuses"] == {"finished": 3, "timed_out": 0, "failed": 0}
    assert report["totals"]["calls"] == 9
    assert [run["scenario"] for run in report["runs"]] == ["s0", "s1", "s2"]
    assert [run["transcript"][0]["text"] for run in report["runs"]] == ["Hi from 0", "Hi from 1", "Hi from 2"]
    assert [run["usage"]["input_tokens"] for run in report["runs"]] == [30, 60, 90]
    assert [run["timeout_seconds"] for run in report["runs"]] == [30.0, 30.0, 30.0]
    assert report["runs"][0]["expectations"] == []
    assert "tool_calls" not in report["runs"][0]  # this target keeps no log


def test_a_run_that_fails_and_one_that_times_out_are_results_and_the_others_go_on(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 5)

    async def act(message):
        if message == "Hi from 1":
            raise TargetError("boom")
        if message == "Hi from 3":
            await asyncio.Event().wait()  # never answers
        return "Hello"

    register_target(monkeypatch, make_scenarios(5, timeouts={3: 0.05}), act=act)

    code, results = run_parallel(tmp_path)

    out = capsys.readouterr().out
    report = json.loads(results.read_text(encoding="utf-8"))
    assert code == runner.EXIT_RUNS_DID_NOT_FINISH
    assert [run["status"] for run in report["runs"]] == ["finished", "failed", "finished", "timed_out", "finished"]
    assert report["runs"][1]["error"] == "boom"
    assert report["runs"][3]["outcome"] is None
    assert [m["text"] for m in report["runs"][3]["transcript"]] == ["Hi from 3"]
    # the timeout this run had is its own, not the default of the batch
    assert [run["timeout_seconds"] for run in report["runs"]] == [120.0, 120.0, 120.0, 0.05, 120.0]
    assert "Runs: 5 | finished: 3 | timed_out: 1 | failed: 1" in out
    assert "s1: failed" in out and "- boom" in out
    assert "s3: timed_out" in out


def test_the_call_limit_reaches_the_runs(monkeypatch, tmp_path):
    use_scripted_customers(monkeypatch, 2)
    register_target(monkeypatch, make_scenarios(2))

    code, results = run_parallel(tmp_path, "--max-calls", "2")

    report = json.loads(results.read_text(encoding="utf-8"))
    assert code == runner.EXIT_RUNS_DID_NOT_FINISH
    assert [run["status"] for run in report["runs"]] == ["failed", "failed"]
    assert {run["error_type"] for run in report["runs"]} == {"CallLimitError"}


# --- where the scenarios and the results are ---

def test_scenarios_can_come_from_a_file_instead_of_the_target(monkeypatch, tmp_path):
    use_scripted_customers(monkeypatch, 2)
    register_target(monkeypatch, scenarios=[])  # the target itself has none
    scenario_file = tmp_path / "mine.toml"
    scenario_file.write_text(
        '[s0]\npersona = "<<persona 0>> You are customer 0."\ngoal = "Get a refund."\n\n'
        '[s1]\npersona = "<<persona 1>> You are customer 1."\ngoal = "Get a refund."\n',
        encoding="utf-8",
    )

    code, results = run_parallel(tmp_path, "--scenario-file", str(scenario_file))

    report = json.loads(results.read_text(encoding="utf-8"))
    assert code == 0
    assert [run["scenario"] for run in report["runs"]] == ["s0", "s1"]


def test_without_a_results_option_the_file_goes_to_the_results_folder(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 1)
    register_target(monkeypatch, make_scenarios(1))
    monkeypatch.chdir(tmp_path)

    code = runner.main(["parallel", "fake"])

    [written] = (tmp_path / "results").glob("fake-*.json")
    assert code == 0
    assert json.loads(written.read_text(encoding="utf-8"))["target"] == "fake"
    assert f"Results written to {runner.RESULTS_FOLDER / written.name}" in capsys.readouterr().out


def test_the_limits_have_defaults_and_can_be_changed(monkeypatch, tmp_path):
    use_fake(monkeypatch, ScriptedInteractions())
    register_target(monkeypatch, make_scenarios(1))
    seen = []

    async def spy(target_module, scenarios, **kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(runner, "run_in_parallel", spy)

    run_parallel(tmp_path)
    run_parallel(tmp_path, "--max-parallel", "3", "--timeout", "7.5", "--max-calls", "11")

    assert (seen[0]["max_parallel"], seen[0]["default_timeout"], seen[0]["max_calls"]) == (
        runner.DEFAULT_MAX_PARALLEL, runner.DEFAULT_TIMEOUT_SECONDS, config.DEFAULT_CALLS_PER_RUN
    )
    assert (seen[1]["max_parallel"], seen[1]["default_timeout"], seen[1]["max_calls"]) == (3, 7.5, 11)
    assert runner.DEFAULT_MAX_PARALLEL == 5 and runner.DEFAULT_TIMEOUT_SECONDS == 120


# --- problems are one clear line ---

def test_a_missing_api_key_stops_the_batch_before_any_run_starts(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(agent_loop, "_get_client", lambda: config.get_api_key())
    module = register_target(monkeypatch, make_scenarios(5))

    code, results = run_parallel(tmp_path)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith(f"error: Environment variable {config.API_KEY_ENV_VAR} is not set")
    assert captured.err.count("error:") == 1  # one error, not one for each of 5 runs
    assert "Running" not in captured.out
    assert module.opened == 0
    assert not results.exists()


def test_a_target_without_scenarios_is_refused(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, ScriptedInteractions())
    register_target(monkeypatch, scenarios=[])

    code, results = run_parallel(tmp_path)

    assert code == 1
    assert "no scenarios to run on target 'fake'" in capsys.readouterr().err
    assert not results.exists()


def test_an_unknown_target_names_the_ones_there_are(capsys, tmp_path):
    code, _ = run_parallel(tmp_path, target="nosuch")

    err = capsys.readouterr().err
    assert code == 1
    assert "Unknown target 'nosuch'" in err and "shop" in err


def test_a_wrong_scenario_file_is_one_line(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, ScriptedInteractions())
    register_target(monkeypatch, make_scenarios(1))
    broken = tmp_path / "broken.toml"
    broken.write_text("this is = = not toml", encoding="utf-8")

    code, results = run_parallel(tmp_path, "--scenario-file", str(broken))

    assert code == 1
    assert "not valid TOML" in capsys.readouterr().err
    assert not results.exists()


def test_a_results_file_that_cannot_be_written_is_reported_after_the_runs(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 1)
    register_target(monkeypatch, make_scenarios(1))
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")

    code = runner.main(["parallel", "fake", "--results", str(blocker / "out.json")])

    captured = capsys.readouterr()
    assert code == 1
    assert "[1/1] s0: finished" in captured.out  # the runs happened
    assert captured.err.startswith("error: Cannot write the results file")


def test_ctrl_c_ends_the_batch_with_a_short_message(monkeypatch, capsys):
    async def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_parallel_command", interrupted)

    code = runner.main(["parallel", "fake"])

    assert code == 130
    assert capsys.readouterr().err.strip() == "Interrupted."


# --- the modes and their options ---

@pytest.mark.parametrize(
    "options, message",
    [
        (["--max-parallel", "0"], "--max-parallel must be at least 1"),
        (["--max-parallel", "-2"], "--max-parallel must be at least 1"),
        (["--timeout", "0"], "--timeout must be greater than 0"),
        (["--timeout", "-5"], "--timeout must be greater than 0"),
        (["--max-calls", "0"], "--max-calls must be at least 1"),
    ],
)
def test_limits_that_make_no_sense_are_refused(options, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["parallel", "fake", *options])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "required"),
        (["fake", "refund"], "invalid choice"),  # the old form, without a mode
        (["batch", "fake"], "invalid choice"),
        (["single", "fake", "refund", "--max-parallel", "2"], "unrecognized arguments"),
        (["single", "fake", "refund", "--results", "out.json"], "unrecognized arguments"),
        (["parallel", "fake", "refund"], "unrecognized arguments"),  # a name belongs to `single`
        (["single", "fake"], "required"),  # `single` needs a scenario
    ],
)
def test_each_mode_accepts_only_its_own_arguments(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        runner.main(argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- the exit code ---

def test_the_exit_codes_are_different_for_each_kind_of_ending():
    """A script can tell a batch that could not start from one that ran but did not all finish."""
    codes = {0, runner.EXIT_COULD_NOT_RUN, runner.EXIT_RUNS_DID_NOT_FINISH, runner.EXIT_INTERRUPTED}

    assert len(codes) == 4
    assert runner.EXIT_COULD_NOT_RUN == 1
    assert runner.EXIT_RUNS_DID_NOT_FINISH == 3  # 2 is argparse's, for a command line it cannot use


def test_a_batch_in_which_every_run_finished_exits_with_0_whatever_the_customers_verdict(monkeypatch, tmp_path):
    """`gave_up` and `cut_off` are what the customer decided, for the judge to weigh. They are not a failure of the batch."""
    use_scripted_customers(monkeypatch, 2)
    register_target(monkeypatch, make_scenarios(2))

    code, results = run_parallel(tmp_path)

    report = json.loads(results.read_text(encoding="utf-8"))
    assert {run["outcome"]["kind"] for run in report["runs"]} == {"gave_up"}
    assert code == 0


def test_a_batch_with_a_run_that_timed_out_exits_with_3_and_still_writes_the_file(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 2)

    async def act(message):
        if message == "Hi from 1":
            await asyncio.Event().wait()  # never answers
        return "Hello"

    register_target(monkeypatch, make_scenarios(2, timeouts={1: 0.05}), act=act)

    code, results = run_parallel(tmp_path)

    assert code == 3
    assert json.loads(results.read_text(encoding="utf-8"))["statuses"] == {"finished": 1, "timed_out": 1, "failed": 0}
    assert "Runs: 2 | finished: 1 | timed_out: 1 | failed: 0" in capsys.readouterr().out


def test_a_key_that_is_refused_by_the_server_fails_every_run_and_exits_with_3(monkeypatch, tmp_path):
    """The key is only checked to be set. A wrong one shows as failed runs, and the exit code must say so."""
    use_fake(monkeypatch, ScriptedInteractions(*[FakeHttpError(401) for _ in range(3)]))
    register_target(monkeypatch, make_scenarios(3))

    code, results = run_parallel(tmp_path)

    report = json.loads(results.read_text(encoding="utf-8"))
    assert report["statuses"] == {"finished": 0, "timed_out": 0, "failed": 3}
    assert {run["error_type"] for run in report["runs"]} == {"ModelError"}
    assert code == 3


def test_a_batch_that_could_not_run_exits_with_1(capsys, tmp_path):
    code, _ = run_parallel(tmp_path, target="nosuch")

    assert code == 1


def test_a_single_run_still_exits_with_0_when_it_ends(monkeypatch):
    """`single` is one conversation for a person to read. There is no batch to judge by its exit code."""
    use_scripted_customers(monkeypatch, 1)
    register_target(monkeypatch, make_scenarios(1))

    assert runner.main(["single", "fake", "s0"]) == 0


# --- several scenario files ---

def write_scenarios(path, *indexes):
    """A scenario file with the scenarios s<i> that `use_scripted_customers` has scripts for."""
    path.write_text(
        "".join(
            f'[s{i}]\npersona = "<<persona {i}>> You are customer {i}."\ngoal = "Get a refund."\n\n'
            for i in indexes
        ),
        encoding="utf-8",
    )
    return str(path)


def test_scenarios_of_several_files_are_played_together_in_the_order_of_the_files(monkeypatch, tmp_path):
    use_scripted_customers(monkeypatch, 3)
    register_target(monkeypatch, scenarios=[])
    first = write_scenarios(tmp_path / "first.toml", 2)
    second = write_scenarios(tmp_path / "second.toml", 0, 1)

    code, results = run_parallel(tmp_path, "--scenario-file", first, "--scenario-file", second)

    report = json.loads(results.read_text(encoding="utf-8"))
    assert code == 0
    assert [run["scenario"] for run in report["runs"]] == ["s2", "s0", "s1"]


def test_the_target_can_come_after_the_scenario_file_option(monkeypatch, tmp_path):
    """The option takes one file, so it does not swallow the target that follows it."""
    use_scripted_customers(monkeypatch, 2)
    register_target(monkeypatch, scenarios=[])
    first = write_scenarios(tmp_path / "first.toml", 0)
    second = write_scenarios(tmp_path / "second.toml", 1)
    results = tmp_path / "out.json"

    code = runner.main([
        "parallel", "--results", str(results), "--scenario-file", first, "--scenario-file", second, "fake",
    ])

    report = json.loads(results.read_text(encoding="utf-8"))
    assert code == 0
    assert report["target"] == "fake"
    assert [run["scenario"] for run in report["runs"]] == ["s0", "s1"]


def test_a_second_file_needs_the_option_again(tmp_path, capsys):
    """`--scenario-file a b` is not a list of two files. The `b` is left over, and the command line is refused."""
    first = write_scenarios(tmp_path / "first.toml", 0)
    second = write_scenarios(tmp_path / "second.toml", 1)

    with pytest.raises(SystemExit) as exit_info:
        runner.main(["parallel", "fake", "--scenario-file", first, second])

    assert exit_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_files_that_have_a_scenario_of_the_same_name_are_refused_before_anything_runs(monkeypatch, capsys, tmp_path):
    use_fake(monkeypatch, ScriptedInteractions())
    module = register_target(monkeypatch, scenarios=[])
    first = write_scenarios(tmp_path / "first.toml", 0)
    second = write_scenarios(tmp_path / "second.toml", 0)

    code, results = run_parallel(tmp_path, "--scenario-file", first, "--scenario-file", second)

    err = capsys.readouterr().err
    assert code == 1
    assert "'s0' is in both" in err and "first.toml" in err and "second.toml" in err
    assert module.opened == 0
    assert not results.exists()


@pytest.mark.parametrize(
    "option",
    ["--scenario", "--scenarios", "--scenario-f", "--max", "--max-par", "--time", "--result"],
)
def test_an_option_name_is_not_guessed_from_its_beginning(option, capsys):
    """`--scenario` could be taken for `--scenario-file`, and `--max` for either of two limits."""
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["parallel", "fake", option, "1"])

    assert exit_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_the_single_mode_does_not_guess_option_names_either(capsys):
    with pytest.raises(SystemExit) as exit_info:
        runner.main(["single", "fake", "refund", "--max", "5"])

    assert exit_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
