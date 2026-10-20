from dataclasses import FrozenInstanceError

import pytest

from evaluator.scenario import Scenario


def make_scenario(**overrides):
    """A scenario with placeholder text; `overrides` replace single fields."""
    fields = {"name": "test", "persona": "test persona", "goal": "test goal"}
    return Scenario(**{**fields, **overrides})


def test_max_messages_defaults_to_8():
    """A scenario that does not say how long it may run gets 8 messages."""
    assert make_scenario().max_messages == 8


def test_max_messages_of_one_is_allowed():
    """One message is the smallest conversation that makes sense."""
    assert make_scenario(max_messages=1).max_messages == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_max_messages_must_be_positive(bad):
    """A scenario that may send no messages is rejected, and the error names the value."""
    with pytest.raises(ValueError, match=f"max_messages must be >= 1, got {bad}"):
        make_scenario(max_messages=bad)


def test_scenario_is_frozen():
    """A scenario cannot be changed after it is built."""
    scenario = make_scenario()

    with pytest.raises(FrozenInstanceError):
        scenario.name = "new name"


def test_expectations_default_to_none():
    """A scenario without written expectations has an empty tuple, so it can still be run."""
    assert make_scenario().expectations == ()


def test_expectations_are_kept_in_order():
    """The sentences are stored as given."""
    scenario = make_scenario(expectations=("must not offer a refund", "must explain why"))

    assert scenario.expectations == ("must not offer a refund", "must explain why")


def test_a_single_string_is_not_taken_for_a_list_of_expectations():
    """Iterating a bare string would give one 'expectation' per letter, so it is refused."""
    with pytest.raises(TypeError, match="tuple"):
        make_scenario(expectations="must not offer a refund")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_expectation_is_refused(blank):
    """An empty sentence cannot be checked against anything."""
    with pytest.raises(ValueError, match="blank"):
        make_scenario(expectations=("must be polite", blank))


def test_timeout_defaults_to_the_runs_own():
    """A scenario that does not name a timeout leaves it to the run."""
    assert make_scenario().timeout_seconds is None


@pytest.mark.parametrize("seconds", [1, 0.5, 120])
def test_a_positive_timeout_is_allowed(seconds):
    assert make_scenario(timeout_seconds=seconds).timeout_seconds == seconds


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_timeout_must_be_positive(bad):
    """A scenario that may take no time cannot run, and the error names the value."""
    with pytest.raises(ValueError, match=f"timeout_seconds must be > 0, got {bad}"):
        make_scenario(timeout_seconds=bad)
