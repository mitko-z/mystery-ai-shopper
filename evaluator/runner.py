"""Run scenarios against a target with a simulated customer.

`run_scenario` lets a customer play one scenario and records the conversation.
`run_in_isolation` does it on a target of its own and reports how it went, and
`run_in_parallel` does it for many scenarios at once. `build_report` and
`write_report` make the results file. The command line is in `evaluator.cli`.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from core.tool_recorder import ToolCall
from core.usage import Usage, track_usage
from evaluator.customer import SimulatedCustomer
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.scenario import Scenario
from evaluator.target import Target, ToolCallSource

logger = logging.getLogger(__name__)

RESULTS_FOLDER = Path("results")


class Speaker(StrEnum):
    """Who said a message. The value is how the speaker is named in the printed conversation."""

    CUSTOMER = "Customer"
    AGENT = "Agent"


@dataclass(frozen=True)
class Message:
    """One message of the conversation."""

    number: int
    speaker: Speaker
    text: str

    def to_dict(self) -> dict[str, object]:
        """The message as a plain dict, ready for JSON.

        The speaker is a stable lowercase key, `customer` or `agent`, not the
        text used when the conversation is printed.
        """
        return {"number": self.number, "speaker": self.speaker.name.lower(), "text": self.text}


@dataclass
class Transcript:
    """What was said in a run, and how it ended.

    The caller creates it and `run_scenario` fills it in. Because the caller
    owns it, what was said before a failure or a timeout is still there
    afterwards. `outcome` stays None until the run has ended by itself.
    """

    messages: list[Message] = field(default_factory=list)
    outcome: Outcome | None = None


class RunStatus(StrEnum):
    """What happened to a run. It is not the customer's outcome: a run that
    `finished` has an outcome, the other two do not.
    """

    FINISHED = "finished"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass
class RunResult:
    """Everything a run left behind, whether it finished or not.

    Attributes:
        scenario: The scenario that was played.
        status: What happened to the run.
        transcript: What was said. Also kept when the run timed out or failed.
        usage: The model calls and tokens of this run alone.
        started_at: When the run began, in UTC.
        duration_seconds: How long the run took.
        timeout_seconds: The timeout the run had: the scenario's own, or else the
            default of the batch. It says why a run that ended after 1 second was
            not given the default one.
        error: Why the run did not finish. None for a finished run.
        error_type: The class name of the error, e.g. `TargetError`.
        tool_calls: The tool calls the target's agent made, for a target that keeps
            such a log. None means the target keeps none, which is not the same as
            an empty list: that means the agent called no tools.
    """

    scenario: Scenario
    status: RunStatus
    transcript: Transcript
    usage: Usage
    started_at: datetime
    duration_seconds: float
    timeout_seconds: float
    error: Optional[str] = None
    error_type: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    @property
    def outcome(self) -> Optional[Outcome]:
        """How the customer ended the conversation. None unless the run finished."""
        return self.transcript.outcome if self.status is RunStatus.FINISHED else None

    def to_dict(self) -> dict[str, Any]:
        """The result as a plain dict, ready for JSON.

        The scenario's expectations, persona and goal are included, so a judge can
        read the result without the scenario file. `tool_calls` is left out for a
        target that keeps no log.
        """
        data: dict[str, Any] = {
            "scenario": self.scenario.name,
            "persona": self.scenario.persona,
            "goal": self.scenario.goal,
            "expectations": list(self.scenario.expectations),
            "status": self.status.value,
            "error": self.error,
            "error_type": self.error_type,
            "outcome": self.outcome.to_dict() if self.outcome is not None else None,
            "started_at": self.started_at.isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "timeout_seconds": self.timeout_seconds,
            "usage": self.usage.to_dict(),
            "transcript": [message.to_dict() for message in self.transcript.messages],
        }
        if self.tool_calls is not None:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return data


async def run_scenario(
    target: Target,
    scenario: Scenario,
    transcript: Transcript,
    on_message: Optional[Callable[[Message], None]] = None,
) -> Outcome:
    """Let a simulated customer play `scenario` against `target`, recording it in `transcript`.

    The customer writes, the target answers, and so on until the customer calls
    `finish`, or has sent `scenario.max_messages` messages. After the last
    answer the customer still gets one turn to finish, so a conversation that
    ends well on the last answer is not reported as cut off. The customer's
    goodbye is part of the transcript but is never sent to the target.

    Messages are added to `transcript` as they happen. If this raises, or is
    cancelled by a timeout, the transcript keeps what was said so far and its
    outcome stays None. When the run ends by itself, the outcome is both
    returned and stored in `transcript.outcome`.

    Args:
        target: The system to talk to.
        scenario: Who the customer is and what they want.
        transcript: Where the conversation and its outcome are recorded. Pass
            an empty one.
        on_message: Called with each message as soon as it exists, to show the
            conversation while it runs.

    Returns:
        How the conversation ended.

    Raises:
        TargetError: If the target fails.
        CustomerError, agent_loop.TurnLimitError, config.MissingAPIKeyError and
            model errors: If the simulated customer fails.
    """
    customer = SimulatedCustomer(scenario)

    def add(speaker: Speaker, text: str) -> None:
        message = Message(len(transcript.messages) + 1, speaker, text.strip())
        transcript.messages.append(message)
        if on_message is not None:
            on_message(message)

    reply = None
    for sent in range(scenario.max_messages + 1):
        text = await customer.next_message(reply)
        if customer.outcome is not None:
            if text.strip():
                add(Speaker.CUSTOMER, text)
            transcript.outcome = customer.outcome
            return customer.outcome
        if sent == scenario.max_messages:
            break  # the customer had its last chance to finish, and did not
        add(Speaker.CUSTOMER, text)
        reply = await target.send(text)
        add(Speaker.AGENT, reply)

    transcript.outcome = Outcome(
        OutcomeKind.CUT_OFF,
        f"The conversation reached the limit of {scenario.max_messages} messages.",
    )
    return transcript.outcome


def _read_tool_calls(target: Target) -> Optional[list[ToolCall]]:
    """The target's log of tool calls, or None if it keeps none or cannot be read.

    Reading the log is an extra: when it fails, the run still has its result.
    """
    if not isinstance(target, ToolCallSource):
        return None
    try:
        return list(target.tool_calls())
    except Exception:
        logger.warning("The target's tool-call log could not be read.", exc_info=True)
        return None


async def run_in_isolation(
    target_module: ModuleType,
    scenario: Scenario,
    *,
    default_timeout: float,
    max_calls: int,
    on_message: Optional[Callable[[Message], None]] = None,
) -> RunResult:
    """Play `scenario` on a target of its own, and report how it went. Never raises.

    The run is isolated in both directions. Nothing leaks in: it opens its own
    target, counts its own model calls, and has its own timeout. Nothing escapes:
    a timeout or any failure ends only this run and becomes its status, and
    the transcript, the cost and the tool log are kept. So many runs can go
    side by side, and one that breaks does not stop the others.

    Only cancellation passes through, so that Ctrl+C still stops the program.

    Args:
        target_module: A module of the `targets` package, with `open_target()`.
        scenario: Who the customer is and what they want.
        default_timeout: Seconds the conversation may take, unless the scenario
            names a timeout of its own.
        max_calls: The most model calls the run may make.
        on_message: Called with each message as soon as it exists.

    Returns:
        The result of the run: finished, timed out or failed.
    """
    timeout = scenario.timeout_seconds if scenario.timeout_seconds is not None else default_timeout
    transcript = Transcript()
    tool_calls = None
    status = RunStatus.FINISHED
    error = error_type = None
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    limit = asyncio.timeout(timeout)

    with track_usage(max_calls) as usage:
        try:
            async with limit:
                async with target_module.open_target() as target:
                    try:
                        await run_scenario(target, scenario, transcript, on_message)
                    finally:
                        tool_calls = _read_tool_calls(target)
        except TimeoutError as e:
            # A TimeoutError can also come from the target, e.g. a socket. It is
            # our timeout only if the limit says it ran out.
            if limit.expired():
                status = RunStatus.TIMED_OUT
                error = f"The run took longer than its timeout of {timeout:g} seconds."
            else:
                status = RunStatus.FAILED
                error = str(e) or type(e).__name__
            error_type = type(e).__name__
        except Exception as e:
            status = RunStatus.FAILED
            error = str(e) or type(e).__name__
            error_type = type(e).__name__

    return RunResult(
        scenario=scenario,
        status=status,
        transcript=transcript,
        usage=usage,
        started_at=started_at,
        duration_seconds=time.perf_counter() - started,
        timeout_seconds=timeout,
        error=error,
        error_type=error_type,
        tool_calls=tool_calls,
    )


async def run_in_parallel(
        target_module: ModuleType,
        scenarios: list[Scenario],
        *,
        max_parallel: int,
        default_timeout: float,
        max_calls: int,
        on_finish: Optional[Callable[[RunResult], None]] = None) -> list[RunResult]:
    """Play all `scenarios`, several at a time, and report how each one went.

    Every scenario runs in isolation (see `run_in_isolation`), so a run that fails
    or times out only ends itself: that is its result, with that status, and the
    other runs go on. Because failures are results, they are not raised.

    At most `max_parallel` runs are active at once. The others wait for a free
    slot, and the wait does not count against their timeout. A model call that
    is being retried after a pause keeps its slot on purpose: while the API is
    rate-limiting us, that stops more runs from adding to the load.

    Args:
        target_module: A module of the `targets` package, with `open_target()`.
            Each run opens its own target from it.
        scenarios: The scenarios to play.
        max_parallel: The most runs active at the same time. At least 1.
        default_timeout: Seconds a conversation may take, unless its scenario
            names a timeout of its own.
        max_calls: The most model calls one run may make.
        on_finish: Called with the result of each run as soon as that run is
            over, in the order the runs finish. A callback that raises is logged
            and ignored, so reporting can never lose a result.

    Returns:
        One result per scenario, in the order of `scenarios` and not in the
        order the runs finished.

    Raises:
        ValueError: If `max_parallel` is less than 1.
    """
    if max_parallel < 1:
        raise ValueError(f"Max parallel runs must be >= 1, got {max_parallel}")

    semaphore = asyncio.Semaphore(max_parallel)

    async def worker(scenario: Scenario) -> RunResult:
        async with semaphore:
            run_result = await run_in_isolation(
                target_module,
                scenario,
                default_timeout=default_timeout,
                max_calls=max_calls)
        if on_finish is not None:
            try:
                on_finish(run_result)
            except Exception:
                logger.exception("The on_finish callback failed for scenario %r. Its result is kept.", scenario.name)
        return run_result

    return await asyncio.gather(*(worker(scenario) for scenario in scenarios))


def count_statuses(results: Sequence[RunResult]) -> dict[str, int]:
    """How many runs ended with each status. Every status is there, also with 0."""
    counts = {status.value: 0 for status in RunStatus}
    for result in results:
        counts[result.status.value] += 1
    return counts


def build_report(
    target_name: str,
    results: Sequence[RunResult],
    *,
    settings: dict[str, Any],
    started_at: datetime,
    duration_seconds: float,
    totals: Usage,
) -> dict[str, Any]:
    """The content of the results file, as a plain dict ready for JSON.

    Args:
        target_name: The target the scenarios were played on.
        results: The result of each run, in scenario order.
        settings: The limits the batch ran with.
        started_at: When the batch began, in UTC.
        duration_seconds: How long the whole batch took.
        totals: What the whole batch cost.
    """
    return {
        "target": target_name,
        "started_at": started_at.isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "settings": settings,
        "statuses": count_statuses(results),
        "totals": totals.to_dict(),
        "runs": [result.to_dict() for result in results],
    }


class ResultsFileError(Exception):
    """Raised when the results file cannot be written."""


def default_results_path(target_name: str, started_at: datetime) -> Path:
    """Where the results of a batch go unless told otherwise, e.g. `results/shop-20261021-101500.json`."""
    return RESULTS_FOLDER / f"{target_name}-{started_at:%Y%m%d-%H%M%S}.json"


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write the report as readable UTF-8 JSON, making the folder if it is missing.

    The file is written under another name first and then moved into place, so
    an interrupted write never leaves half a file where the results should be.

    Raises:
        ResultsFileError: If the file cannot be written.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError as e:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ResultsFileError(f"Cannot write the results file {path}: {e.strerror or e}") from e
