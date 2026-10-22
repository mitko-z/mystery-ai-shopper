"""Load scenarios from a TOML file.

Each scenario is a table named after the scenario:

    [refund_o1002]
    persona = "You are Alex, ..."
    goal = "Get a full refund for order O1002."
    max_messages = 8             # optional, default 8
    timeout_seconds = 120        # optional, default: the run's timeout
    expectations = [             # optional
        "The agent must not refund an order that is only shipped.",
    ]

A wrong file is refused with a message that names the file and the scenario, so a
typo such as `expectation` is found at once and not silently ignored.
"""

import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluator.scenario import Scenario

_REQUIRED_KEYS = ("persona", "goal")
_OPTIONAL_KEYS = ("max_messages", "timeout_seconds", "expectations")


class ScenarioFileError(Exception):
    """Raised when a scenario file cannot be read or has something wrong in it."""


def load_scenarios(path: str | Path) -> dict[str, Scenario]:
    """Read all scenarios of a TOML file.

    Args:
        path: The file to read.

    Returns:
        The scenarios by name, in the order of the file. The name of a scenario is
        the name of its table.

    Raises:
        ScenarioFileError: If the file cannot be read, is not valid TOML, holds no
            scenarios, or a scenario in it is wrong.
    """
    path = Path(path)
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except OSError as e:
        raise ScenarioFileError(f"Cannot read the scenario file {path}: {e.strerror or e}") from e
    except tomllib.TOMLDecodeError as e:
        raise ScenarioFileError(f"The scenario file {path} is not valid TOML: {e}") from e
    if not data:
        raise ScenarioFileError(f"The scenario file {path} holds no scenarios.")
    return {name: _build_scenario(path, name, table) for name, table in data.items()}


def load_scenario_files(paths: Iterable[str | Path]) -> dict[str, Scenario]:
    """Read the scenarios of several TOML files into one dict.

    Args:
        paths: The files to read, in order.

    Returns:
        The scenarios by name: those of the first file, then those of the next, and
        so on.

    Raises:
        ScenarioFileError: If a file is wrong (see `load_scenarios`), or if two files
            have a scenario of the same name. Silently keeping one of them would run
            a scenario other than the one that was asked for.
    """
    scenarios: dict[str, Scenario] = {}
    source: dict[str, Path] = {}
    for path in map(Path, paths):
        for name, scenario in load_scenarios(path).items():
            if name in scenarios:
                raise ScenarioFileError(
                    f"The scenario {name!r} is in both {source[name]} and {path}. "
                    "Scenario names must be different."
                )
            scenarios[name] = scenario
            source[name] = path
    return scenarios


def _build_scenario(path: Path, name: str, table: Any) -> Scenario:
    """Check one table of the file and build its scenario."""

    def fail(problem: str) -> ScenarioFileError:
        return ScenarioFileError(f"{path}: scenario {name!r}: {problem}")

    if not isinstance(table, dict):
        raise fail("expected a table, written as [name] followed by its keys.")
    unknown = sorted(set(table) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        allowed = ", ".join((*_REQUIRED_KEYS, *_OPTIONAL_KEYS))
        raise fail(f"unknown key(s): {', '.join(unknown)}. The keys are: {allowed}.")
    for key in _REQUIRED_KEYS:
        if key not in table:
            raise fail(f"the key {key!r} is missing.")
        if not isinstance(table[key], str) or not table[key].strip():
            raise fail(f"{key!r} must be text that is not blank.")
    if "max_messages" in table and not _is_int(table["max_messages"]):
        raise fail("'max_messages' must be a whole number.")
    if "timeout_seconds" in table and not _is_number(table["timeout_seconds"]):
        raise fail("'timeout_seconds' must be a number.")
    expectations = table.get("expectations", [])
    if not isinstance(expectations, list) or not all(isinstance(item, str) for item in expectations):
        raise fail("'expectations' must be a list of sentences.")

    options = {key: table[key] for key in ("max_messages", "timeout_seconds") if key in table}
    try:
        return Scenario(
            name=name,
            persona=table["persona"].strip(),
            goal=table["goal"].strip(),
            expectations=tuple(expectations),
            **options,
        )
    except (TypeError, ValueError) as e:
        raise fail(str(e)) from e


def _is_int(value: Any) -> bool:
    """True for a whole number. A bool is not one, although Python says so."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """True for a whole or decimal number, not for a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
