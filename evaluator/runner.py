"""Run a scenario against a target and print the conversation.

    uv run python -m evaluator.runner <target> <scenario>

`<target>` is a module in the `targets` package, e.g. `shop`. It provides
`open_target()`, an async context manager that yields a `Target`, and
`SCENARIOS`, a dict of scenarios by name.
"""

import argparse
import asyncio
import importlib
import pkgutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from types import ModuleType
from typing import Optional

import httpx

from core import agent_loop, config
from core.usage import Usage, track_usage
from evaluator.customer import CustomerError, SimulatedCustomer
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.scenario import Scenario
from evaluator.target import Target, TargetError

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
