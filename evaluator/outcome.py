from dataclasses import dataclass
from enum import StrEnum


class OutcomeKind(StrEnum):
    """Kinds of outcomes for a scenario evaluation."""

    GOAL_ACHIEVED = "goal_achieved"
    GAVE_UP = "gave_up"
    CUT_OFF = "cut_off"


@dataclass(frozen=True)
class Outcome:
    """The result of running a scenario.

    Attributes:
        kind: The kind of outcome, e.g. `goal_achieved`.
        reason: A description of why the outcome occurred.
    """

    kind: OutcomeKind
    reason: str

    def to_dict(self) -> dict[str, str]:
        """The outcome as a plain dict, ready for JSON."""
        return {"kind": self.kind.value, "reason": self.reason}
