from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """A single scenario for evaluation.

    Attributes:
        name: Short name of the scenario, e.g. "refund".
        persona: Description of the shopper's persona, including what they bought
            and what facts they know.
        goal: The shopper's secret goal, e.g. "get a refund for a delivered order".
        max_messages: Maximum number of messages to send before cutting off the
            conversation. Default is 8.
    """

    name: str
    persona: str
    goal: str
    max_messages: int = 8

    def __post_init__(self):
        """Validate the scenario's attributes."""
        if self.max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {self.max_messages}")
