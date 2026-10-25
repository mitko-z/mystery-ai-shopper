"""Compare the judge's verdicts with the ground-truth key, and write the agreement table.

The key (`evaluations/<name>.ground-truth.json`) holds our own verdict for each
expectation of a results file. The judge's file (`...judge-v1.json`, see
`evaluator.judge`) holds the judge's. A row of the table puts the two side by side.

`cannot_tell` is a verdict of the judge, never of the key. Where the expectation needs
something the judge cannot see (`needs_tool_log` in the key), it is not a mistake: it
is the honest answer. Anywhere else it is a miss.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from evaluator.judge import Verdict

UNCLEAR = "unclear"  # a verdict that only the key can have: the expectation can be read two ways
KEY_VERDICTS = (Verdict.PASS.value, Verdict.FAIL.value, UNCLEAR)
JUDGE_VERDICTS = tuple(verdict.value for verdict in Verdict)

# What the `Agree?` column says.
YES = "yes"
NO = "no"
CANNOT_TELL = Verdict.CANNOT_TELL.value
NO_VERDICT = "no verdict"  # the judge reported none, or did not judge this expectation
NOT_COMPARED = "n/a"  # the key itself is unclear, so there is nothing to agree with

CAUSE_TODO = "TODO"
MAX_EXPECTATION_CHARS = 90


class AgreementError(Exception):
    """Raised when the key and the judge's file cannot be compared, or the table cannot be written."""


@dataclass(frozen=True)
class Row:
    """One expectation, as the key and the judge saw it.

    Attributes:
        scenario: Name of the scenario.
        expectation_number: Position of the expectation in the scenario, from 1.
        expectation: The sentence that was judged.
        needs_tool_log: Whether the key says the verdict depends on something the
            judge cannot see: tool calls, tool results or the agent's instruction.
        covered_by: The number of another expectation of the scenario that says the
            same, for a duplicate row. None otherwise.
        key: Our verdict: `pass`, `fail` or `unclear`.
        judge: The judge's verdict: `pass`, `fail` or `cannot_tell`. None if it
            reported none, or its file has nothing for this expectation.
    """

    scenario: str
    expectation_number: int
    expectation: str
    needs_tool_log: bool
    covered_by: Optional[int]
    key: str
    judge: Optional[str]

    @property
    def agree(self) -> str:
        """`yes`, `no`, `cannot_tell`, `no verdict` or `n/a`, as the `Agree?` column says."""
        if self.key == UNCLEAR:
            return NOT_COMPARED
        if self.judge is None:
            return NO_VERDICT
        if self.judge == CANNOT_TELL:
            return CANNOT_TELL
        return YES if self.judge == self.key else NO

    @property
    def needs_cause(self) -> bool:
        """Whether the table asks for a cause: a disagreement, a missing verdict, or a `cannot_tell` that is a miss."""
        return self.agree in (NO, NO_VERDICT) or (self.agree == CANNOT_TELL and not self.needs_tool_log)


@dataclass(frozen=True)
class Counts:
    """How a set of rows came out.

    Attributes:
        rows: How many rows.
        yes: The judge gave the key's verdict.
        no: The judge gave the opposite verdict.
        cannot_tell_expected: The judge could not tell, and the expectation needs the tool log.
        cannot_tell_missed: The judge could not tell, although the conversation was enough.
        not_compared: Rows with no judge verdict, or with an unclear key.
    """

    rows: int = 0
    yes: int = 0
    no: int = 0
    cannot_tell_expected: int = 0
    cannot_tell_missed: int = 0
    not_compared: int = 0


def load_verdicts(path: str | Path, what: str) -> Mapping[str, Any]:
    """Read the key or the judge's file.

    Args:
        path: The JSON file.
        what: What the file is, for the messages: `key` or `judge file`.

    Raises:
        AgreementError: If the file cannot be read, is not UTF-8 JSON, or holds no
            object with a list of `verdicts`.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise AgreementError(f"Cannot read the {what} {path}: {e.strerror or e}") from e
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AgreementError(f"The {what} {path} is not valid UTF-8 JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
        raise AgreementError(f"The {what} {path} has no list of verdicts.")
    return data


def _entries(data: Mapping[str, Any], what: str, allowed: Sequence[str], *, judge: bool) -> dict[tuple[str, int], Mapping[str, Any]]:
    """The verdict entries by (scenario, expectation number), checked.

    Raises:
        AgreementError: If an entry lacks a field, holds a verdict that `allowed` does
            not name, or is there twice.
    """
    entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in data["verdicts"]:
        fields_ok = (
            isinstance(entry, Mapping)
            and isinstance(entry.get("scenario"), str)
            and isinstance(entry.get("expectation_number"), int)
            and isinstance(entry.get("expectation"), str)
            and (judge or isinstance(entry.get("needs_tool_log"), bool))
        )
        if not fields_ok:
            raise AgreementError(f"An entry of the {what} lacks its scenario, expectation number, expectation text"
                                 + ("." if judge else " or needs_tool_log."))
        name = (entry["scenario"], entry["expectation_number"])
        verdict = entry.get("verdict")
        if not (verdict in allowed or (judge and verdict is None)):
            raise AgreementError(f"The {what} gives {name[0]} #{name[1]} the verdict {verdict!r}; "
                                 f"it can be one of: {', '.join(allowed)}.")
        if name in entries:
            raise AgreementError(f"The {what} has {name[0]} #{name[1]} twice.")
        entries[name] = entry
    return entries


def compare(key: Mapping[str, Any], judge: Mapping[str, Any]) -> list[Row]:
    """Put each verdict of the key beside the judge's verdict for the same expectation.

    Args:
        key: The content of the key file.
        judge: The content of the judge's file.

    Returns:
        One row per entry of the key, in its order. An expectation the judge has no
        verdict for (its judgement failed, or its run was skipped) is a row with
        `judge` None, so that it shows in the table and is not hidden.

    Raises:
        AgreementError: If a file is malformed, or the judge's file holds an expectation
            that the key does not, or the two files word the same expectation differently.
            Then they were not made from the same results file.
    """
    key_entries = _entries(key, "key", KEY_VERDICTS, judge=False)
    judge_entries = _entries(judge, "judge file", JUDGE_VERDICTS, judge=True)

    unknown = sorted(set(judge_entries) - set(key_entries))
    if unknown:
        name = unknown[0]
        raise AgreementError(
            f"The judge file has a verdict for {name[0]} #{name[1]}, which the key does not cover. "
            "Were they made from the same results file?"
        )

    rows = []
    for name, entry in key_entries.items():
        judged = judge_entries.get(name)
        if judged is not None and judged["expectation"] != entry["expectation"]:
            raise AgreementError(
                f"{name[0]} #{name[1]} is worded differently in the key and in the judge file. "
                "Were they made from the same results file?"
            )
        rows.append(Row(
            scenario=name[0],
            expectation_number=name[1],
            expectation=entry["expectation"],
            needs_tool_log=entry["needs_tool_log"],
            covered_by=entry.get("covered_by"),
            key=entry["verdict"],
            judge=judged["verdict"] if judged is not None else None,
        ))
    return rows


def compare_files(key_path: str | Path, judge_path: str | Path) -> tuple[list[Row], Mapping[str, Any]]:
    """Read the key and the judge's file and compare them.

    Returns:
        The rows (see `compare`), and the `judge` block of the judge's file: its model,
        prompt version and mode. It is empty if the file has none.

    Raises:
        AgreementError: See `load_verdicts` and `compare`.
    """
    key = load_verdicts(key_path, "key")
    judge = load_verdicts(judge_path, "judge file")
    info = judge.get("judge")
    return compare(key, judge), info if isinstance(info, Mapping) else {}


def _count(rows: Sequence[Row]) -> Counts:
    """How the rows came out."""
    return Counts(
        rows=len(rows),
        yes=sum(row.agree == YES for row in rows),
        no=sum(row.agree == NO for row in rows),
        cannot_tell_expected=sum(row.agree == CANNOT_TELL and row.needs_tool_log for row in rows),
        cannot_tell_missed=sum(row.agree == CANNOT_TELL and not row.needs_tool_log for row in rows),
        not_compared=sum(row.agree in (NO_VERDICT, NOT_COMPARED) for row in rows),
    )


def summarize(rows: Sequence[Row]) -> list[tuple[str, Counts]]:
    """The rows counted four ways.

    Returns:
        (label, counts) for all rows; for the rows without the duplicates (a duplicate
        is judged twice for the same facts, which would count them double); and, of
        those, the rows where the transcript is enough and the rows that need the log.
        The last two show where the judge works without the log, and what the log adds.
    """
    unique = [row for row in rows if row.covered_by is None]
    return [
        ("All rows", _count(rows)),
        ("Without the duplicate rows", _count(unique)),
        ("of which the transcript is enough", _count([row for row in unique if not row.needs_tool_log])),
        ("of which the tool log is needed", _count([row for row in unique if row.needs_tool_log])),
    ]


def format_summary(rows: Sequence[Row]) -> str:
    """The summary as plain lines, for the terminal."""
    return "\n".join(
        f"{label}: {c.rows} | yes {c.yes} | no {c.no} | cannot_tell expected {c.cannot_tell_expected}, "
        f"missed {c.cannot_tell_missed} | not compared {c.not_compared}"
        for label, c in summarize(rows)
    )


def _cell(text: str, limit: Optional[int] = None) -> str:
    """Text made safe for a cell of a Markdown table: one line, no bare `|`, and shortened if `limit` is given."""
    text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text.replace("|", "\\|")


def render_markdown(
    rows: Sequence[Row],
    *,
    key_name: str,
    judge_name: str,
    judge_info: Mapping[str, Any],
) -> str:
    """The agreement table as Markdown, to be read by a person.

    A row that needs a cause says `TODO` in its last cell, for the person to replace.

    Args:
        rows: The rows, as `compare` returns them.
        key_name: The name of the key file, for the heading.
        judge_name: The name of the judge's file.
        judge_info: The `judge` block of the judge's file: model, prompt version and mode.
    """
    judge_text = (
        f"model {judge_info.get('model', '?')}, rules {judge_info.get('prompt_version', '?')}, "
        f"mode {judge_info.get('mode', '?')}"
    )
    lines = [
        f"# Agreement: {judge_name}",
        "",
        f"- Key: `{key_name}`",
        f"- Judge: `{judge_name}` ({judge_text})",
        "",
        "`Agree?` is `yes` when the judge gave the key's verdict and `no` when it gave the opposite one. "
        "`cannot_tell` means the judge said the evidence is not enough: that is expected where the "
        "expectation needs the tool log (`Needs log` is yes), and a miss anywhere else. "
        f"`no verdict` means the judge reported none. A row that needs a cause says `{CAUSE_TODO}` until someone writes it.",
        "",
        "| Run | # | Expectation | Needs log | Key | Judge | Agree? | Cause |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        expectation = _cell(row.expectation, MAX_EXPECTATION_CHARS)
        if row.covered_by is not None:
            expectation += f" (covered by #{row.covered_by})"
        cells = [
            _cell(row.scenario),
            str(row.expectation_number),
            expectation,
            "yes" if row.needs_tool_log else "no",
            row.key,
            row.judge if row.judge is not None else "-",
            row.agree,
            CAUSE_TODO if row.needs_cause else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Summary",
        "",
        "| | Rows | yes | no | cannot_tell (expected) | cannot_tell (missed) | not compared |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, c in summarize(rows):
        lines.append(
            f"| {label} | {c.rows} | {c.yes} | {c.no} | {c.cannot_tell_expected} | {c.cannot_tell_missed} | {c.not_compared} |"
        )
    return "\n".join(lines) + "\n"


def default_out_path(judge_path: str | Path) -> Path:
    """Where the table goes unless told otherwise: beside the judge's file, named after it.

    `evaluations/shop-20261023-185334.judge-v1.json` gives
    `evaluations/shop-20261023-185334.agreement-v1.md`.
    """
    judge_path = Path(judge_path)
    return judge_path.with_name(judge_path.name.replace(".judge-", ".agreement-")).with_suffix(".md")


def write_table(path: str | Path, text: str, *, overwrite: bool = False) -> None:
    """Write the table as UTF-8, making the folder if it is missing.

    The causes are written into the table by hand, so a table that exists is not
    overwritten unless `overwrite` says so.

    Raises:
        AgreementError: If the file exists and `overwrite` is false, or cannot be written.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise AgreementError(f"{path} exists already, and may hold causes that were written by hand.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise AgreementError(f"Cannot write the table {path}: {e.strerror or e}") from e
