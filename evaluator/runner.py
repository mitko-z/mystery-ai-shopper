"""Run a scenario against a target and print the conversation.

    uv run python -m evaluator.runner <target> <scenario>

`<target>` is a module in the `targets` package, e.g. `shop`. It provides
`open_target()`, an async context manager that yields a `Target`, and
`SCENARIOS`, a dict of scenarios by name.
"""

import argparse
import asyncio
import importlib
import logging
import pkgutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import ModuleType
from typing import Any, Optional

import httpx

from core import agent_loop, config
from core.tool_recorder import ToolCall
from core.usage import Usage, track_usage
from evaluator.customer import CustomerError, SimulatedCustomer
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.scenario import Scenario
from evaluator.scenario_file import ScenarioFileError
from evaluator.target import Target, TargetError, ToolCallSource

logger = logging.getLogger(__name__)

TARGETS_PACKAGE = "targets"

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


def format_header(scenario: Scenario) -> str:
    """The lines that open a printed run: who the customer is and what they want."""
    return (
        f"Scenario: {scenario.name}\n"
        f"Persona: {scenario.persona.strip()}\n"
        f"Secret goal: {scenario.goal.strip()}\n"
    )


def format_message(message: Message) -> str:
    """One numbered line of the conversation."""
    return f"{message.number}. {message.speaker}: {message.text}"


def format_outcome(outcome: Outcome) -> str:
    """The line that closes a printed run."""
    return f"Outcome: {outcome.kind.value} - {outcome.reason}"


def format_usage(usage: Usage) -> str:
    """The line that reports what a run cost: model calls and tokens.

    Calls that cannot be in the token counts are named: cancelled calls, which
    never got a response, and calls whose response carried no usage report.
    """
    line = (
        f"Model calls: {usage.calls} | Tokens: input {usage.input_tokens:,}, "
        f"output {usage.output_tokens:,}, thought {usage.thought_tokens:,}, "
        f"total {usage.total_tokens:,}"
    )
    if usage.calls_cancelled:
        line += f" | Calls cancelled (tokens unknown): {usage.calls_cancelled}"
    if usage.calls_without_usage:
        line += f" | Calls without a usage report: {usage.calls_without_usage} (tokens are incomplete)"
    return line


class UsageError(Exception):
    """Raised when the command line names a target or scenario that does not exist."""


def load_target(name: str) -> ModuleType:
    """Import the module of target `name` from the targets package.

    Raises:
        UsageError: If there is no such target; the message lists the ones there are.
    """
    package = importlib.import_module(TARGETS_PACKAGE)
    module_name = f"{TARGETS_PACKAGE}.{name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name != module_name:
            raise  # the target exists but is broken
        available = ", ".join(sorted(m.name for m in pkgutil.iter_modules(package.__path__))) or "none"
        raise UsageError(f"Unknown target {name!r}. Available: {available}.") from e


def get_scenario(target_module: ModuleType, name: str) -> Scenario:
    """Look up scenario `name` in a target module.

    Raises:
        UsageError: If there is no such scenario; the message lists the ones there are.
    """
    scenarios = target_module.SCENARIOS
    if name not in scenarios:
        raise UsageError(f"Unknown scenario {name!r}. Available: {', '.join(sorted(scenarios))}.")
    return scenarios[name]


async def _run(target_name: str, scenario_name: str, max_calls: int) -> None:
    """Run one scenario and print the conversation as it happens.

    The outcome, if the run reached one, and what the run cost are printed at the
    end, also when the run fails.
    """
    target_module = load_target(target_name)
    scenario = get_scenario(target_module, scenario_name)
    print(format_header(scenario))
    transcript = Transcript()
    with track_usage(max_calls) as usage:
        try:
            async with target_module.open_target() as target:
                await run_scenario(target, scenario, transcript, on_message=lambda m: print(format_message(m)))
        finally:
            print()
            if transcript.outcome is not None:
                print(format_outcome(transcript.outcome))
            print(format_usage(usage))


def main(argv: Optional[list[str]] = None) -> int:
    """Command line entry point.

    Returns:
        0 if the run finished (whatever the outcome), 1 if it could not be run.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evaluator.runner",
        description="Run a scenario with a simulated customer against a target.",
    )
    parser.add_argument("target", help=f"name of a module in the `{TARGETS_PACKAGE}` package, e.g. shop")
    parser.add_argument("scenario", help="name of one of the target's scenarios, e.g. refund_o1002")
    parser.add_argument("--max-calls", help="most model calls one run may make (default: 60)", type=int, default=config.DEFAULT_CALLS_PER_RUN)
    args = parser.parse_args(argv)

    if args.max_calls < 1:
        parser.error("--max-calls must be at least 1")

    # Model text can hold characters the Windows console codepage cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        asyncio.run(_run(args.target, args.scenario, args.max_calls))
    except (
        UsageError,
        ScenarioFileError,
        TargetError,
        CustomerError,
        config.MissingAPIKeyError,
        agent_loop.TurnLimitError,
        agent_loop.ModelError,
        httpx.HTTPError,
        agent_loop.CallLimitError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
