from dataclasses import dataclass
from typing import Optional


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
        expectations: What the target must and must not do in this scenario,
            one plain sentence each, e.g. "must not offer a refund for a shipped
            order". They are the ground truth a judge compares the transcript
            with. Neither the simulated customer nor the target ever sees them.
        timeout_seconds: How long the whole conversation may take, in seconds.
            None means the run's default.
    """

    name: str
    persona: str
    goal: str
    max_messages: int = 8
    expectations: tuple[str, ...] = ()
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        """Validate the scenario's attributes."""
        if self.max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {self.max_messages}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if isinstance(self.expectations, str):
            raise TypeError("expectations must be a tuple of sentences, not a single string")
        if any(not expectation.strip() for expectation in self.expectations):
            raise ValueError("an expectation must not be blank")
