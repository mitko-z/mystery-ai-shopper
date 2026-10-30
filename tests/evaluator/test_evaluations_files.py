"""The controls and their key, which are made by hand, and the code and scenarios they have to agree with."""

import json
from pathlib import Path

import pytest

from evaluator import agreement, judge
from evaluator.scenario_file import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS = ROOT / "evaluations"

# name -> (the controls, their key, the scenario file whose words they use)
CONTROL_SETS = {
    "first batch": (
        EVALUATIONS / "shop-20261023-185334.controls.json",
        EVALUATIONS / "shop-20261023-185334.controls.ground-truth.json",
        ROOT / "scenarios" / "shop.toml",
    ),
    "second batch": (
        EVALUATIONS / "shop-20261029-133551.controls.json",
        EVALUATIONS / "shop-20261029-133551.controls.ground-truth.json",
        ROOT / "scenarios" / "shop_batch2.toml",
    ),
}
BATCH2_RESULTS = ROOT / "results" / "shop-20261029-133551.json"
BATCH2_KEY = EVALUATIONS / "shop-20261029-133551.ground-truth.json"

control_sets = pytest.mark.parametrize("controls_path, key_path, scenario_file", CONTROL_SETS.values(), ids=CONTROL_SETS)


def controls(path):
    runs, skipped = judge.select_runs(judge.load_results(path))
    assert skipped == []  # every control is a finished run with expectations
    return runs


@control_sets
def test_every_expectation_of_the_controls_has_a_verdict_in_their_key_in_the_same_words(controls_path, key_path, scenario_file):
    expected = {
        (run["scenario"], number): text
        for run in controls(controls_path)
        for number, text in enumerate(run["expectations"], start=1)
    }

    entries = agreement.load_verdicts(key_path, "key")["verdicts"]

    assert {(e["scenario"], e["expectation_number"]): e["expectation"] for e in entries} == expected


@control_sets
def test_the_controls_use_the_words_of_the_scenario_file(controls_path, key_path, scenario_file):
    """The expectations of a real run come from there, and a control that rewords one tests another sentence."""
    known = {text for scenario in load_scenarios(scenario_file).values() for text in scenario.expectations}

    used = [text for run in controls(controls_path) for text in run["expectations"]]

    assert used and all(text in known for text in used)


@control_sets
def test_the_controls_have_good_agents_to_be_passed_and_violations_to_be_failed(controls_path, key_path, scenario_file):
    """A judge that passes everything, or one that fails every policy sentence, must not look right on these."""
    key = agreement.load_verdicts(key_path, "key")
    rows = agreement.compare(key, {"verdicts": []})

    verdicts = [row.key for row in rows]

    assert 4 <= verdicts.count("pass") <= 6
    assert verdicts.count("fail") >= 1
    assert "unclear" not in verdicts


@control_sets
def test_the_tool_log_of_each_control_is_kept_for_the_judge_that_may_read_it(controls_path, key_path, scenario_file):
    assert all(run["tool_calls"] for run in controls(controls_path))


def test_every_expectation_of_the_second_batch_has_a_verdict_in_its_key_in_the_same_words():
    """The key of a real run is written by hand. The judge file is refused if one of its sentences differs from it."""
    runs, skipped = judge.select_runs(judge.load_results(BATCH2_RESULTS))
    assert skipped == []
    expected = {
        (run["scenario"], number): text
        for run in runs
        for number, text in enumerate(run["expectations"], start=1)
    }

    entries = agreement.load_verdicts(BATCH2_KEY, "key")["verdicts"]

    assert {(e["scenario"], e["expectation_number"]): e["expectation"] for e in entries} == expected


def test_the_second_batch_uses_the_words_of_its_scenario_file():
    known = {text for scenario in load_scenarios(ROOT / "scenarios" / "shop_batch2.toml").values() for text in scenario.expectations}

    used = [text for run in judge.load_results(BATCH2_RESULTS)["runs"] for text in run["expectations"]]

    assert used and all(text in known for text in used)


def test_the_key_of_the_second_batch_can_fail_a_judge_that_passes_everything():
    rows = agreement.compare(agreement.load_verdicts(BATCH2_KEY, "key"), {"verdicts": []})

    verdicts = [row.key for row in rows]

    assert len(rows) == 16
    assert verdicts.count("fail") >= 2
    assert "unclear" not in verdicts  # an unclear row would not be compared, and the criteria count every row


def test_every_judge_file_holds_the_rules_that_its_version_still_has_in_the_code():
    """A version is never changed once judgements were made with it. If its text changes, the files lie about it."""
    files = sorted(EVALUATIONS.rglob("*.judge-*.json"))

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
