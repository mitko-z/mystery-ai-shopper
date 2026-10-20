import json
from dataclasses import FrozenInstanceError

import pytest

from evaluator.outcome import Outcome, OutcomeKind


def test_outcome_keeps_its_kind_and_reason():
    """An outcome holds what it was built with, and its kind equals the plain string."""
    outcome = Outcome(kind=OutcomeKind.GAVE_UP, reason="Customer gave up.")

    assert outcome.kind is OutcomeKind.GAVE_UP
    assert outcome.kind == "gave_up"
    assert outcome.reason == "Customer gave up."


def test_outcome_kind_values():
    """The kinds have the strings that the customer's `finish` tool and the transcript use."""
    assert {kind.value for kind in OutcomeKind} == {"goal_achieved", "gave_up", "cut_off"}


def test_unknown_outcome_kind_is_rejected():
    """Only the known kinds exist, so converting a made-up string fails."""
    with pytest.raises(ValueError):
        OutcomeKind("quit")


def test_outcome_is_frozen():
    """An outcome cannot be changed after it is built."""
    outcome = Outcome(kind=OutcomeKind.GAVE_UP, reason="Customer gave up.")

    with pytest.raises(FrozenInstanceError):
        outcome.kind = OutcomeKind.GOAL_ACHIEVED


def test_outcome_becomes_a_plain_dict():
    """The kind is its plain string, so the dict can be written as JSON."""
    as_dict = Outcome(OutcomeKind.GAVE_UP, "Refused.").to_dict()

    assert as_dict == {"kind": "gave_up", "reason": "Refused."}
    assert type(as_dict["kind"]) is str
    assert json.loads(json.dumps(as_dict)) == as_dict
