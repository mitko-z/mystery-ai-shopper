"""The `agreement` mode of the command line, and the entry point `python -m evaluator`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import agent_loop, config
from evaluator import cli

ROOT = Path(__file__).resolve().parents[2]


def key_entry(verdict="pass", number=1, expectation="The agent must be polite.", needs_tool_log=False):
    return {
        "scenario": "s", "expectation_number": number, "expectation": expectation,
        "verdict": verdict, "evidence": "e", "needs_tool_log": needs_tool_log,
    }


def judge_entry(verdict="pass", number=1, expectation="The agent must be polite."):
    return {"scenario": "s", "expectation_number": number, "expectation": expectation, "verdict": verdict}


def write_files(tmp_path, key_entries, judge_entries, judge_name="shop-1.judge-v1.json"):
    key = tmp_path / "shop-1.ground-truth.json"
    key.write_text(json.dumps({"verdicts": key_entries}), encoding="utf-8")
    judge = tmp_path / judge_name
    judge.write_text(json.dumps({
        "judge": {"model": "m", "prompt_version": "v1", "mode": "transcript"},
        "verdicts": judge_entries,
    }), encoding="utf-8")
    return key, judge


# --- a table ---

def test_the_table_is_written_and_a_summary_names_the_rows_that_need_a_cause(capsys, tmp_path):
    key, judge = write_files(
        tmp_path,
        [key_entry("pass", 1), key_entry("fail", 2, "Another sentence."), key_entry("pass", 3, "Third.", True)],
        [judge_entry("pass", 1), judge_entry("pass", 2, "Another sentence."), judge_entry("cannot_tell", 3, "Third.")],
    )
    out = tmp_path / "table.md"

    code = cli.main(["agreement", str(key), str(judge), "--out", str(out)])

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines[0] == "All rows: 3 | yes 1 | no 1 | cannot_tell expected 1, missed 0 | not compared 0"
    assert "1 row needs a cause: TODO in the table." in lines
    assert lines[-1] == f"Agreement table written to {out}"
    text = out.read_text(encoding="utf-8")
    assert "- Judge: `shop-1.judge-v1.json` (model m, rules v1, mode transcript)" in text
    assert text.count("| TODO |") == 1


def test_several_rows_that_need_a_cause_are_counted_in_the_plural(capsys, tmp_path):
    key, judge = write_files(
        tmp_path,
        [key_entry("pass", 1), key_entry("pass", 2, "Another sentence.")],
        [judge_entry("fail", 1), judge_entry("fail", 2, "Another sentence.")],
    )

    cli.main(["agreement", str(key), str(judge), "--out", str(tmp_path / "table.md")])

    assert "2 rows need a cause: TODO in the table." in capsys.readouterr().out.splitlines()


def test_without_an_out_option_the_table_goes_beside_the_judge_file_named_after_it(capsys, tmp_path):
    key, judge = write_files(tmp_path, [key_entry()], [judge_entry()], judge_name="shop-1.judge-v1-toollog.json")

    code = cli.main(["agreement", str(key), str(judge)])

    assert code == 0
    assert (tmp_path / "shop-1.agreement-v1-toollog.md").is_file()
    assert "need a cause" not in capsys.readouterr().out


def test_the_comparison_needs_no_model_and_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(agent_loop, "_get_client", lambda: config.get_api_key())
    key, judge = write_files(tmp_path, [key_entry()], [judge_entry()])

    assert cli.main(["agreement", str(key), str(judge), "--out", str(tmp_path / "t.md")]) == 0


# --- a table that has causes in it ---

def test_a_table_that_exists_is_kept_unless_overwrite_is_given(capsys, tmp_path):
    """The causes in it are written by hand. Making the table again must not lose them."""
    key, judge = write_files(tmp_path, [key_entry()], [judge_entry()])
    out = tmp_path / "table.md"
    out.write_text("my causes", encoding="utf-8")

    code = cli.main(["agreement", str(key), str(judge), "--out", str(out)])

    assert code == 1
    assert "table.md exists already" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "my causes"

    code = cli.main(["agreement", str(key), str(judge), "--out", str(out), "--overwrite"])

    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("# Agreement:")


# --- problems are one clear line ---

def test_files_that_do_not_belong_together_are_one_clear_line_and_no_table(capsys, tmp_path):
    key, judge = write_files(tmp_path, [key_entry()], [judge_entry(expectation="Another sentence.")])
    out = tmp_path / "table.md"

    code = cli.main(["agreement", str(key), str(judge), "--out", str(out)])

    err = capsys.readouterr().err
    assert code == 1
    assert "worded differently" in err and err.count("error:") == 1
    assert not out.exists()


def test_a_file_that_is_missing_is_one_clear_line(capsys, tmp_path):
    key, _ = write_files(tmp_path, [key_entry()], [judge_entry()])

    code = cli.main(["agreement", str(key), str(tmp_path / "nosuch.json")])

    assert code == 1
    assert "Cannot read the judge file" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["agreement"], "required"),
        (["agreement", "key.json"], "required"),
        (["agreement", "key.json", "judge.json", "extra.json"], "unrecognized arguments"),
        (["agreement", "key.json", "judge.json", "--tool-log"], "unrecognized arguments"),  # an option of `judge`
        (["agreement", "key.json", "judge.json", "--over"], "unrecognized arguments"),  # not a guess at --overwrite
    ],
)
def test_agreement_accepts_only_its_own_arguments(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- the entry point ---

def test_python_dash_m_evaluator_starts_the_command_line_and_names_all_four_commands():
    result = subprocess.run(
        [sys.executable, "-m", "evaluator", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0
    assert result.stdout.startswith("usage: python -m evaluator")
    for command in ("single", "parallel", "judge", "agreement"):
        assert command in result.stdout


def test_python_dash_m_evaluator_exits_with_2_for_a_command_line_it_cannot_use():
    result = subprocess.run(
        [sys.executable, "-m", "evaluator", "nosuch"], cwd=ROOT, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
