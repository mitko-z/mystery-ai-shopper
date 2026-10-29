"""The controls and their key, which are made by hand, and the code and scenarios they have to agree with."""

import json
from pathlib import Path

from evaluator import agreement, judge
from evaluator.scenario_file import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "evaluations" / "shop-20261023-185334.controls.json"
CONTROLS_KEY = ROOT / "evaluations" / "shop-20261023-185334.controls.ground-truth.json"


def controls():
    runs, skipped = judge.select_runs(judge.load_results(CONTROLS))
    assert skipped == []  # every control is a finished run with expectations
    return runs


def test_every_expectation_of_the_controls_has_a_verdict_in_their_key_in_the_same_words():
    expected = {
        (run["scenario"], number): text
        for run in controls()
        for number, text in enumerate(run["expectations"], start=1)
    }

    entries = agreement.load_verdicts(CONTROLS_KEY, "key")["verdicts"]

    assert {(e["scenario"], e["expectation_number"]): e["expectation"] for e in entries} == expected


def test_the_controls_use_the_words_of_the_scenario_file():
    """The expectations of a real run come from there, and a control that rewords one tests another sentence."""
    known = {text for scenario in load_scenarios(ROOT / "scenarios" / "shop.toml").values() for text in scenario.expectations}

    used = [text for run in controls() for text in run["expectations"]]

    assert used and all(text in known for text in used)


def test_the_controls_have_good_agents_to_be_passed_and_one_violation_to_be_failed():
    """A judge that passes everything, or one that fails every policy sentence, must not look right on these."""
    key = agreement.load_verdicts(CONTROLS_KEY, "key")
    rows = agreement.compare(key, {"verdicts": []})

    verdicts = [row.key for row in rows]

    assert 4 <= verdicts.count("pass") <= 6
    assert verdicts.count("fail") >= 1
    assert "unclear" not in verdicts


def test_the_tool_log_of_each_control_is_kept_for_the_judge_that_may_read_it():
    assert all(run["tool_calls"] for run in controls())


def test_every_judge_file_holds_the_rules_that_its_version_still_has_in_the_code():
    """A version is never changed once judgements were made with it. If its text changes, the files lie about it."""
    files = sorted((ROOT / "evaluations").rglob("*.judge-*.json"))

    versions = set()
    for path in files:
        info = json.loads(path.read_text(encoding="utf-8"))["judge"]
        versions.add(info["prompt_version"])
        assert info["rules"] == {
            "common": judge._COMMON_RULES[info["prompt_version"]],
            "transcript": judge._VIEW_TRANSCRIPT,
            "tool_log": judge._VIEW_TOOL_LOG,
        }, path.name

    assert "v1" in versions  # the files that the comparison with v2 is made against
