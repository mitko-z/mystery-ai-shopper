"""The `single` and `parallel` modes with a target read from a config file (`--target-config`)."""

import json

import httpx
import pytest

from core import usage as usage_module
from core.usage import Usage
from evaluator import cli
from evaluator.target_config import HttpTargetConfig
from tests.evaluator.helpers import use_scripted_customers
from tests.evaluator.test_cli_parallel import register_target, write_scenarios
from tests.fakes import ScriptedInteractions, use_fake

SECRET = "s3cr3t-value-123"

CONFIG = """
name = "acme-support"
base_url = "https://agent.example.com"
path = "/v1/chat"

[auth]
header = "Authorization"
env = "ACME_TOKEN"
prefix = "Bearer "

[request]
body = { session = "{conversation_id}", input = { text = "{message}" } }

[response]
reply = "data.reply"
"""


@pytest.fixture(autouse=True)
def fresh_grand_total(monkeypatch):
    """The batch prints the grand total of the process, so each test starts it from zero."""
    monkeypatch.setattr(usage_module, "_grand_total", Usage())


@pytest.fixture
def requests(monkeypatch):
    """Answer every request of a config target in the process, and keep the requests.

    The endpoint answers `Hello from Acme` at `data.reply`, so a run never goes to the network.
    """
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": {"reply": "Hello from Acme"}})

    original = HttpTargetConfig.open_target
    monkeypatch.setattr(
        HttpTargetConfig, "open_target", lambda self: original(self, transport=httpx.MockTransport(handler))
    )
    return seen


def write_config(tmp_path, text=CONFIG):
    path = tmp_path / "acme.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- parallel ---

def test_a_batch_runs_on_the_endpoint_of_the_config(monkeypatch, capsys, tmp_path, requests):
    monkeypatch.setenv("ACME_TOKEN", SECRET)
    use_scripted_customers(monkeypatch, 2)
    config = write_config(tmp_path)
    scenarios = write_scenarios(tmp_path / "scenarios.toml", 0, 1)
    results = tmp_path / "out.json"

    code = cli.main([
        "parallel", "--target-config", config, "--scenario-file", scenarios, "--results", str(results),
    ])

    out = capsys.readouterr().out
    report = json.loads(results.read_text(encoding="utf-8"))
    assert code == 0
    assert out.splitlines()[0] == "Running 2 scenarios on acme-support, at most 5 at a time."
    assert report["target"] == "acme-support"
    assert report["settings"]["target_config"] == config
    assert [run["scenario"] for run in report["runs"]] == ["s0", "s1"]
    assert [run["transcript"][1]["text"] for run in report["runs"]] == ["Hello from Acme"] * 2
    assert all("tool_calls" not in run for run in report["runs"])  # a black box: cannot tell

    assert len(requests) == 2
    assert {str(request.url) for request in requests} == {"https://agent.example.com/v1/chat"}
    assert all(request.headers["Authorization"] == f"Bearer {SECRET}" for request in requests)
    bodies = [json.loads(request.content) for request in requests]
    assert sorted(body["input"]["text"] for body in bodies) == ["Hi from 0", "Hi from 1"]
    assert bodies[0]["session"] != bodies[1]["session"]  # every run is a conversation of its own


def test_the_secret_is_neither_printed_nor_written(monkeypatch, capsys, tmp_path, requests):
    monkeypatch.setenv("ACME_TOKEN", SECRET)
    use_scripted_customers(monkeypatch, 1)
    results = tmp_path / "out.json"

    cli.main([
        "parallel", "--target-config", write_config(tmp_path),
        "--scenario-file", write_scenarios(tmp_path / "scenarios.toml", 0), "--results", str(results),
    ])

    captured = capsys.readouterr()
    assert requests  # the secret was used
    assert SECRET not in captured.out + captured.err
    assert SECRET not in results.read_text(encoding="utf-8")


def test_the_results_file_is_named_after_the_config_target(monkeypatch, tmp_path, requests):
    monkeypatch.setenv("ACME_TOKEN", SECRET)
    use_scripted_customers(monkeypatch, 1)
    config = write_config(tmp_path)
    scenarios = write_scenarios(tmp_path / "scenarios.toml", 0)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["parallel", "--target-config", config, "--scenario-file", scenarios])

    assert code == 0
    assert len(list((tmp_path / "results").glob("acme-support-*.json"))) == 1


def test_a_missing_secret_stops_the_batch_before_any_request(monkeypatch, capsys, tmp_path, requests):
    monkeypatch.delenv("ACME_TOKEN", raising=False)
    use_fake(monkeypatch, ScriptedInteractions())  # no model call may be made
    results = tmp_path / "out.json"

    code = cli.main([
        "parallel", "--target-config", write_config(tmp_path),
        "--scenario-file", write_scenarios(tmp_path / "scenarios.toml", 0), "--results", str(results),
    ])

    err = capsys.readouterr().err
    assert code == cli.EXIT_COULD_NOT_RUN
    assert err.startswith("error: Environment variable ACME_TOKEN is not set.")
    assert len(err.strip().splitlines()) == 1
    assert requests == []
    assert not results.exists()


def test_a_wrong_config_is_one_line_that_names_the_file(monkeypatch, capsys, tmp_path, requests):
    use_fake(monkeypatch, ScriptedInteractions())
    config = write_config(tmp_path, 'base_url = "https://a.example"\nbase-url = "typo"\n')

    code = cli.main([
        "parallel", "--target-config", config, "--scenario-file", write_scenarios(tmp_path / "s.toml", 0),
    ])

    err = capsys.readouterr().err.strip()
    assert code == cli.EXIT_COULD_NOT_RUN
    assert err.startswith(f"error: {config}: unknown key(s): base-url.")
    assert len(err.splitlines()) == 1
    assert requests == []


def test_an_endpoint_that_answers_with_an_error_fails_the_run_and_exits_with_3(monkeypatch, tmp_path):
    use_scripted_customers(monkeypatch, 1)
    original = HttpTargetConfig.open_target
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="agent down"))
    monkeypatch.setattr(HttpTargetConfig, "open_target", lambda self: original(self, transport=transport))
    config = write_config(tmp_path, 'base_url = "https://a.example"\n')
    results = tmp_path / "out.json"

    code = cli.main([
        "parallel", "--target-config", config,
        "--scenario-file", write_scenarios(tmp_path / "s.toml", 0), "--results", str(results),
    ])

    [run] = json.loads(results.read_text(encoding="utf-8"))["runs"]
    assert code == cli.EXIT_SOME_FAILED
    assert run["status"] == "failed"
    assert run["error"] == "The target answered 500: agent down"


# --- single ---

def test_a_single_run_plays_one_scenario_of_the_file_on_the_endpoint(monkeypatch, capsys, tmp_path, requests):
    monkeypatch.setenv("ACME_TOKEN", SECRET)
    use_scripted_customers(monkeypatch, 2)

    code = cli.main([
        "single", "--target-config", write_config(tmp_path),
        "--scenario-file", write_scenarios(tmp_path / "scenarios.toml", 0, 1), "s1",
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "Scenario: s1" in out
    assert "1. Customer: Hi from 1" in out
    assert "2. Agent: Hello from Acme" in out
    assert SECRET not in out
    assert len(requests) == 1


def test_a_single_run_names_the_scenarios_of_the_file_when_given_an_unknown_one(monkeypatch, capsys, tmp_path, requests):
    monkeypatch.setenv("ACME_TOKEN", SECRET)

    code = cli.main([
        "single", "--target-config", write_config(tmp_path),
        "--scenario-file", write_scenarios(tmp_path / "scenarios.toml", 0, 1), "nope",
    ])

    assert code == cli.EXIT_COULD_NOT_RUN
    assert "Unknown scenario 'nope'. Available: s0, s1." in capsys.readouterr().err
    assert requests == []


def test_a_target_module_can_take_its_scenarios_from_a_file_in_a_single_run_too(monkeypatch, capsys, tmp_path):
    use_scripted_customers(monkeypatch, 1)
    register_target(monkeypatch, scenarios=[])

    code = cli.main(["single", "fake", "s0", "--scenario-file", write_scenarios(tmp_path / "s.toml", 0)])

    assert code == 0
    assert "2. Agent: Hello" in capsys.readouterr().out


# --- the command line ---

@pytest.mark.parametrize(
    "argv, message",
    [
        (["parallel"], "a target is required: `parallel <target>`, or --target-config FILE"),
        (["single", "refund"], "a target is required: `single <target> <scenario>`, or --target-config FILE"),
        (["parallel", "fake", "--target-config", "a.toml", "--scenario-file", "s.toml"],
         "give a target or --target-config, not both"),
        (["single", "fake", "refund", "--target-config", "a.toml", "--scenario-file", "s.toml"],
         "give a target or --target-config, not both"),
        (["parallel", "--target-config", "a.toml"], "--target-config needs --scenario-file"),
        (["single", "--target-config", "a.toml", "refund"], "--target-config needs --scenario-file"),
        (["parallel", "--target", "a.toml"], "unrecognized arguments"),  # the name is not guessed
    ],
)
def test_a_command_line_that_does_not_name_one_target_is_refused(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_in_a_single_run_options_go_before_the_target_or_after_the_scenario(capsys):
    """With the target optional, argparse takes a lone word before an option for the scenario.

    So `single fake --max-calls 5 refund` is refused, where it was accepted before
    `--target-config` existed. It runs nothing.
    """
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["single", "fake", "--max-calls", "5", "refund"])

    assert exit_info.value.code == 2
    assert "unrecognized arguments: refund" in capsys.readouterr().err
