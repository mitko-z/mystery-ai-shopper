"""Count the model calls and tokens a run uses."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

from google.genai.interactions import Usage as ReportedUsage


@dataclass
class Usage:
    """What a run has spent so far: model calls and the tokens the API reported.

    The totals only grow. Tokens the API leaves out of a report count as 0, so
    they never make a field `None`.

    Attributes:
        calls: Model calls that got a response.
        input_tokens: Tokens sent to the model.
        output_tokens: Tokens the model wrote as its answer.
        thought_tokens: Tokens the model spent thinking.
        total_tokens: Tokens in all, as the API counts them.
        calls_without_usage: Calls whose response had no usage report at all. The
            token totals miss these, so a low total is only trustworthy when this is 0.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    calls_without_usage: int = 0

    def add(self, reported: Optional[ReportedUsage]) -> None:
        """Count one model call that got a response.

        Args:
            reported: The usage report of that response, `Interaction.usage` of
                the SDK. None if the response carried none.
        """
        self.calls += 1
        if reported is None:
            self.calls_without_usage += 1
            return
        # Each count is Optional[int] in the SDK: a missing one is 0.
        self.input_tokens += reported.total_input_tokens or 0
        self.output_tokens += reported.total_output_tokens or 0
        self.thought_tokens += reported.total_thought_tokens or 0
        self.total_tokens += reported.total_tokens or 0


_current: ContextVar[Optional[Usage]] = ContextVar("current_usage", default=None)
_grand_total = Usage()


@contextmanager
def track_usage() -> Generator[Usage]:
    """Start counting a run: calls made in the block are added to the yielded Usage.

    Enter it inside the coroutine of each run, not around the code that starts
    the runs. A new task copies the context it is created in, so runs started
    under one `track_usage()` would all share its Usage.

    Yields:
        A fresh Usage for this run. It stays readable after the block ends.
    """
    usage = Usage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def record(reported: Optional[ReportedUsage]) -> None:
    """Count one model call into the grand total and, if a run is tracked, into its Usage.

    Args:
        reported: The usage report of the response, `Interaction.usage` of the
            SDK. None if the response carried none.
    """
    _grand_total.add(reported)
    current = _current.get()
    if current is not None:
        current.add(reported)


def grand_total() -> Usage:
    """Everything counted in this process, whether or not a run was tracked.

    Returns:
        The live total, not a copy.
    """
    return _grand_total