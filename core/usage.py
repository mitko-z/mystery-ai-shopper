"""Count the model calls and tokens a run uses."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Optional

from google.genai.interactions import Usage as ReportedUsage

from core import config


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
        calls_cancelled: Calls abandoned before a response came, e.g. by a timeout.
            They are likely billed but their tokens are unknown, so the token
            totals miss these too. Not part of `calls`.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    calls_without_usage: int = 0
    calls_cancelled: int = 0

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

    def add_cancelled(self) -> None:
        """Count one model call that was abandoned before a response came."""
        self.calls_cancelled += 1

    def to_dict(self) -> dict[str, int]:
        """The counts as a plain dict, ready for JSON."""
        return asdict(self)


_current: ContextVar[Optional[Usage]] = ContextVar("current_usage", default=None)
_current_max_calls: ContextVar[Optional[int]] = ContextVar("current_max_calls", default=None)
_grand_total = Usage()


@contextmanager
def track_usage(max_calls: int = config.DEFAULT_CALLS_PER_RUN) -> Generator[Usage]:
    """Start counting a run: calls made in the block are added to the yielded Usage.

    Enter it inside the coroutine of each run, not around the code that starts
    the runs. A new task copies the context it is created in, so runs started
    under one `track_usage()` would all share its Usage.

    Args:
        max_calls: The most model calls the run may make. Cancelled calls count
            too. Once the run has made that many, the next call raises
            `agent_loop.CallLimitError`.

    Yields:
        A fresh Usage for this run. It stays readable after the block ends.
    """
    usage = Usage()
    max_calls_token = _current_max_calls.set(max_calls)
    usage_token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(usage_token)
        _current_max_calls.reset(max_calls_token)


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


def record_cancelled() -> None:
    """Count one model call that was abandoned before a response came.

    It goes into the grand total and, if a run is tracked, into its Usage.
    """
    _grand_total.add_cancelled()
    current = _current.get()
    if current is not None:
        current.add_cancelled()


def grand_total() -> Usage:
    """Everything counted in this process, whether or not a run was tracked.

    Returns:
        The live total, not a copy.
    """
    return _grand_total


def current_call_limit_reached() -> bool:
    """Say whether the current run has used up the model calls it may make.

    Calls that got a response and cancelled calls both count.

    Returns:
        True if the run has made as many calls as it may. False if it has not,
        or if no run is tracked: then there is no limit.
    """
    usage = _current.get()
    limit = _current_max_calls.get()
    if usage is None or limit is None:
        return False
    return usage.calls + usage.calls_cancelled >= limit


def current_max_calls() -> Optional[int]:
    """The most model calls the current run may make.

    Returns:
        The limit, or None if no run is tracked.
    """
    return _current_max_calls.get()
