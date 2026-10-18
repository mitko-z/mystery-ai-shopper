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
