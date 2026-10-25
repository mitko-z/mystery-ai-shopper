"""The agreement table: the key and the judge's verdicts side by side, and how they are counted."""

import json
from pathlib import Path

import pytest

from evaluator.agreement import (
    Counts,
    AgreementError,
    Row,
    compare,
    compare_files,
    default_out_path,
    format_summary,
    load_verdicts,
    render_markdown,
    summarize,
    write_table,
)

KEY_IN_THE_REPO = Path(__file__).resolve().parents[2] / "evaluations" / "shop-20261023-185334.ground-truth.json"


def key_entry(verdict="pass", scenario="s", number=1, expectation="The agent must be polite.", needs_tool_log=False, **extra):
    return {
        "scenario": scenario, "expectation_number": number, "expectation": expectation,
        "verdict": verdict, "evidence": "e", "needs_tool_log": needs_tool_log, **extra,
    }


def judge_entry(verdict="pass", scenario="s", number=1, expectation="The agent must be polite."):
    return {"scenario": scenario, "expectation_number": number, "expectation": expectation, "verdict": verdict}


def row(key="pass", judge="pass", needs_tool_log=False, covered_by=None, scenario="s", number=1, expectation="The agent must be polite."):
    return Row(scenario, number, expectation, needs_tool_log, covered_by, key, judge)


# --- one row ---

@pytest.mark.parametrize(
    "key, judge, needs_tool_log, agree, needs_cause",
    [
        ("pass", "pass", False, "yes", False),
        ("fail", "fail", True, "yes", False),
        ("pass", "fail", False, "no", True),
        ("fail", "pass", True, "no", True),
        ("pass", "cannot_tell", True, "cannot_tell", False),   # the honest answer: the log is needed
        ("pass", "cannot_tell", False, "cannot_tell", True),   # a miss: the conversation was enough
        ("pass", None, False, "no verdict", True),
        ("unclear", "pass", False, "n/a", False),
        ("unclear", "cannot_tell", False, "n/a", False),
        ("unclear", None, False, "n/a", False),
    ],
)
def test_a_row_says_whether_the_judge_agrees_and_whether_it_needs_a_cause(key, judge, needs_tool_log, agree, needs_cause):
    r = row(key, judge, needs_tool_log)

    assert r.agree == agree
    assert r.needs_cause is needs_cause


# --- comparing the two files ---

def test_each_verdict_of_the_key_gets_a_row_in_its_order_with_the_judges_verdict_beside_it():
    key = {"verdicts": [
        key_entry("fail", "b", 1, needs_tool_log=True),
        key_entry("pass", "a", 2, covered_by=1),
        key_entry("pass", "a", 1),
    ]}
    judge = {"verdicts": [judge_entry("pass", "a", 1), judge_entry("fail", "b", 1)]}

    rows = compare(key, judge)

    assert [(r.scenario, r.expectation_number, r.key, r.judge) for r in rows] == [
        ("b", 1, "fail", "fail"), ("a", 2, "pass", None), ("a", 1, "pass", "pass"),
    ]
    assert [r.needs_tool_log for r in rows] == [True, False, False]
    assert [r.covered_by for r in rows] == [None, 1, None]


def test_an_expectation_the_judge_has_no_verdict_for_is_a_row_and_not_hidden():
    """Its judgement failed (verdict null), or its run was skipped (no entry)."""
    key = {"verdicts": [key_entry(number=1), key_entry(number=2), key_entry(number=3)]}
    judge = {"verdicts": [judge_entry(number=1), judge_entry(None, number=2)]}

    rows = compare(key, judge)

    assert [r.judge for r in rows] == ["pass", None, None]
    assert [r.agree for r in rows] == ["yes", "no verdict", "no verdict"]


@pytest.mark.parametrize(
    "key, judge, message",
    [
        ({"verdicts": [key_entry()]}, {"verdicts": [judge_entry(), judge_entry(number=2)]}, "s #2, which the key does not cover"),
        ({"verdicts": [key_entry()]}, {"verdicts": [judge_entry(expectation="Another sentence.")]}, "s #1 is worded differently"),
        ({"verdicts": [key_entry(), key_entry()]}, {"verdicts": []}, "key has s #1 twice"),
        ({"verdicts": [key_entry()]}, {"verdicts": [judge_entry(), judge_entry()]}, "judge file has s #1 twice"),
        ({"verdicts": [key_entry("cannot_tell")]}, {"verdicts": []}, "the key gives s #1 the verdict 'cannot_tell'|key gives s #1 the verdict"),
        ({"verdicts": [key_entry()]}, {"verdicts": [judge_entry("unclear")]}, "judge file gives s #1 the verdict 'unclear'"),
        ({"verdicts": [key_entry()]}, {"verdicts": [judge_entry("maybe")]}, "judge file gives s #1 the verdict 'maybe'"),
        ({"verdicts": [key_entry(None)]}, {"verdicts": []}, "key gives s #1 the verdict None"),
        ({"verdicts": [{**key_entry(), "needs_tool_log": "yes"}]}, {"verdicts": []}, "entry of the key lacks"),
        ({"verdicts": [{"scenario": "s"}]}, {"verdicts": []}, "entry of the key lacks"),
        ({"verdicts": [key_entry()]}, {"verdicts": [{"scenario": "s", "verdict": "pass"}]}, "entry of the judge file lacks"),
        ({"verdicts": ["oops"]}, {"verdicts": []}, "entry of the key lacks"),
    ],
)
def test_files_that_do_not_belong_together_or_are_malformed_are_refused_with_a_clear_message(key, judge, message):
    with pytest.raises(AgreementError, match=message):
        compare(key, judge)


def test_the_wording_is_only_compared_where_the_judge_has_a_verdict():
    """A run the judge skipped has no entry at all, and that is not a different wording."""
    rows = compare({"verdicts": [key_entry()]}, {"verdicts": []})

    assert rows[0].judge is None


# --- reading the files ---

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_two_files_are_read_and_compared_and_the_judge_block_comes_with_them(tmp_path):
    key = write_json(tmp_path / "key.json", {"verdicts": [key_entry()]})
    judge = write_json(tmp_path / "judge.json", {
        "judge": {"model": "m", "prompt_version": "v1", "mode": "transcript"},
        "verdicts": [judge_entry("fail")],
    })

    rows, info = compare_files(key, judge)

    assert [r.agree for r in rows] == ["no"]
    assert info == {"model": "m", "prompt_version": "v1", "mode": "transcript"}


def test_a_judge_file_without_a_judge_block_gives_an_empty_one(tmp_path):
    key = write_json(tmp_path / "key.json", {"verdicts": [key_entry()]})
    judge = write_json(tmp_path / "judge.json", {"verdicts": [judge_entry()]})

    assert compare_files(key, judge)[1] == {}


def test_a_file_that_cannot_be_read_is_one_clear_message(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{nope", encoding="utf-8")
    not_text = tmp_path / "bytes.json"
    not_text.write_bytes(b"\xff\xfe{")
    no_list = write_json(tmp_path / "nolist.json", {"verdicts": "x"})
    a_list = write_json(tmp_path / "list.json", [])

    with pytest.raises(AgreementError, match="Cannot read the key"):
        load_verdicts(tmp_path / "missing.json", "key")
    for broken in (bad_json, not_text):
        with pytest.raises(AgreementError, match="judge file .* is not valid UTF-8 JSON"):
            load_verdicts(broken, "judge file")
    for broken in (no_list, a_list):
        with pytest.raises(AgreementError, match="has no list of verdicts"):
            load_verdicts(broken, "key")


def test_a_file_is_read_as_utf_8(tmp_path):
    """The default encoding of this machine would not read the text of a conversation."""
    path = write_json(tmp_path / "key.json", {"verdicts": [key_entry(expectation="café — Привет")]})

    assert load_verdicts(path, "key")["verdicts"][0]["expectation"] == "café — Привет"


def test_the_key_in_the_repo_can_be_read_by_the_comparison():
    """The loader and the key file are written apart. This is where they meet."""
    rows = compare(load_verdicts(KEY_IN_THE_REPO, "key"), {"verdicts": []})

    assert len(rows) == 18  # four runs of results/shop-20261023-185334.json
    assert {r.key for r in rows} <= {"pass", "fail", "unclear"}
    assert [r.covered_by for r in rows if r.covered_by is not None] == [5]


# --- counting ---

def test_the_rows_are_counted_all_without_duplicates_and_by_whether_the_log_is_needed():
    rows = [
        row("pass", "pass", needs_tool_log=False),                # yes
        row("pass", "fail", needs_tool_log=False),                # no
        row("pass", "cannot_tell", needs_tool_log=False),         # a miss
        row("pass", "cannot_tell", needs_tool_log=True),          # expected
        row("fail", "fail", needs_tool_log=True),                 # yes
        row("fail", "fail", needs_tool_log=True, covered_by=5),   # a duplicate
        row("pass", None, needs_tool_log=False),                  # no verdict
        row("unclear", "pass", needs_tool_log=False),             # no key to agree with
    ]

    summary = dict(summarize(rows))

    assert summary["All rows"] == Counts(rows=8, yes=3, no=1, cannot_tell_expected=1, cannot_tell_missed=1, not_compared=2)
    assert summary["Without the duplicate rows"] == Counts(rows=7, yes=2, no=1, cannot_tell_expected=1, cannot_tell_missed=1, not_compared=2)
    assert summary["of which the transcript is enough"] == Counts(rows=5, yes=1, no=1, cannot_tell_missed=1, not_compared=2)
    assert summary["of which the tool log is needed"] == Counts(rows=2, yes=1, cannot_tell_expected=1)


def test_the_two_halves_add_up_to_the_rows_without_duplicates():
    rows = [row(), row("fail", "pass", True), row("pass", "cannot_tell", True, covered_by=2)]
    summary = dict(summarize(rows))

    halves = summary["of which the transcript is enough"].rows + summary["of which the tool log is needed"].rows
    assert halves == summary["Without the duplicate rows"].rows == 2


def test_the_summary_is_plain_lines_for_the_terminal():
    rows = [row("pass", "pass"), row("pass", "fail", True)]

    assert format_summary(rows).splitlines() == [
        "All rows: 2 | yes 1 | no 1 | cannot_tell expected 0, missed 0 | not compared 0",
        "Without the duplicate rows: 2 | yes 1 | no 1 | cannot_tell expected 0, missed 0 | not compared 0",
        "of which the transcript is enough: 1 | yes 1 | no 0 | cannot_tell expected 0, missed 0 | not compared 0",
        "of which the tool log is needed: 1 | yes 0 | no 1 | cannot_tell expected 0, missed 0 | not compared 0",
    ]


# --- the table ---

INFO = {"model": "gemini-x", "prompt_version": "v1", "mode": "transcript"}


def render(rows):
    return render_markdown(rows, key_name="k.ground-truth.json", judge_name="k.judge-v1.json", judge_info=INFO)


def table_rows(text):
    """The cells of each row of the first table."""
    lines = [line for line in text.split("\n\n## Summary")[0].splitlines() if line.startswith("| ")]
    return [[cell.strip() for cell in line.strip("|").split(" | ")] for line in lines[2:]]  # not the header and the dashes


def test_the_table_names_the_files_and_what_the_judge_was():
    text = render([row()])

    assert text.startswith("# Agreement: k.judge-v1.json\n")
    assert "- Key: `k.ground-truth.json`" in text
    assert "- Judge: `k.judge-v1.json` (model gemini-x, rules v1, mode transcript)" in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_a_judge_block_with_gaps_does_not_break_the_heading():
    text = render_markdown([row()], key_name="k", judge_name="j", judge_info={})

    assert "(model ?, rules ?, mode ?)" in text


def test_each_row_shows_the_two_verdicts_and_asks_for_a_cause_only_where_one_is_needed():
    rows = [
        row("pass", "pass", scenario="a", number=1),
        row("fail", "pass", scenario="a", number=2),
        row("pass", "cannot_tell", True, scenario="b", number=1),
        row("pass", "cannot_tell", False, scenario="b", number=2),
        row("pass", None, scenario="c", number=1),
    ]

    cells = table_rows(render(rows))

    assert [(c[0], c[1], c[3], c[4], c[5], c[6], c[7]) for c in cells] == [
        ("a", "1", "no", "pass", "pass", "yes", ""),
        ("a", "2", "no", "fail", "pass", "no", "TODO"),
        ("b", "1", "yes", "pass", "cannot_tell", "cannot_tell", ""),
        ("b", "2", "no", "pass", "cannot_tell", "cannot_tell", "TODO"),
        ("c", "1", "no", "pass", "-", "no verdict", "TODO"),
    ]


def test_a_duplicate_row_says_which_expectation_covers_it():
    cells = table_rows(render([row(covered_by=5)]))

    assert cells[0][2].endswith("(covered by #5)")


def test_a_long_expectation_is_shortened_and_a_pipe_in_it_cannot_break_the_table():
    long_text = "The agent must not state " + "x" * 200 + " | and more"

    cells = table_rows(render([row(expectation=long_text), row(expectation="a | b")]))

    assert len(cells[0][2]) <= 90 and cells[0][2].endswith("...")
    assert cells[1][2] == "a \\| b"
    assert all(len(c) == 8 for c in cells)


def test_the_table_ends_with_the_summary():
    text = render([row("pass", "pass"), row("fail", "pass", True)])

    summary = text.split("## Summary")[1]
    assert "| All rows | 2 | 1 | 1 | 0 | 0 | 0 |" in summary
    assert "| of which the tool log is needed | 1 | 0 | 1 | 0 | 0 | 0 |" in summary


# --- where the table goes ---

def test_the_table_is_named_after_the_judge_file():
    assert default_out_path("evaluations/shop-20261023-185334.judge-v1.json").as_posix() == (
        "evaluations/shop-20261023-185334.agreement-v1.md"
    )
    assert default_out_path("evaluations/shop-20261023-185334.judge-v1-toollog.json").as_posix() == (
        "evaluations/shop-20261023-185334.agreement-v1-toollog.md"
    )


def test_the_table_is_written_as_utf_8_and_its_folder_is_made(tmp_path):
    path = tmp_path / "new folder" / "table.md"

    write_table(path, "café — Привет\n")

    assert path.read_text(encoding="utf-8") == "café — Привет\n"


def test_a_table_with_causes_written_by_hand_is_not_overwritten_unless_asked(tmp_path):
    path = tmp_path / "table.md"
    path.write_text("my causes", encoding="utf-8")

    with pytest.raises(AgreementError, match="exists already"):
        write_table(path, "new")
    assert path.read_text(encoding="utf-8") == "my causes"

    write_table(path, "new", overwrite=True)
    assert path.read_text(encoding="utf-8") == "new"


def test_a_table_that_cannot_be_written_is_one_clear_message(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")

    with pytest.raises(AgreementError, match="Cannot write the table"):
        write_table(blocker / "table.md", "x")
