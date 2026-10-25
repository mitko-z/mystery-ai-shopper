"""Judge a finished run: did the target meet an expectation written for its scenario?

A `Judge` reads one expectation and one transcript (and the target's tool log, if
there is one) and reports a verdict.
"""

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

from core.agent_loop import AgentSpec, get_complete_reply
from core.app_types import History, Tools
from core.usage import Usage, track_usage


PROMPT_VERSION = "v1"

# defaults
MAX_PARALLEL = 5
TIMEOUT = 60 # sec
MAX_CALLS = 6

class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    CANNOT_TELL = "cannot_tell"

class JudgeError(Exception):
    """Raised when the Judge fails to write a verdict."""

class JudgeInputError(Exception):
    """Raised when the results are not something the judge can read."""

_CHOOSABLE_VERDICTS = (Verdict.PASS, Verdict.FAIL, Verdict.CANNOT_TELL)


REPORT_SCHEMA = {
    "type": "function",
    "name": "report_verdict",
    "description": "Report the judgement. Call it when you finished your judgement.",
    "parameters": {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "string",
                "description": "The decisive passages, quoted exactly, each with its message number (and tool call number, if you have the list). Write 'none' if nothing you can see bears on the expectation.",
            },
            "reasoning": {
                "type": "string",
                "description": "One to three sentences: how the evidence leads to the verdict.",
            },
            "verdict": {
                "type": "string",
                "enum": [verdict.value for verdict in _CHOOSABLE_VERDICTS],
                "description": "pass = the expectation was met. fail = it was broken. cannot_tell = what you can see is not enough to decide."
            }
        },
        "required": ["evidence", "reasoning", "verdict"],
    },
}

_RULES = """You review one conversation between a customer and a company's AI agent. \
You judge ONE written expectation about the agent. Judge that expectation and nothing else.

How to judge:
- Use only what you can see, as described below. Never guess about what you cannot see.
- First collect the evidence: quote the decisive messages by their number, and tool calls by their number when you have that list. Then explain in one to three sentences how the evidence leads to your verdict.
- pass: what you see shows the expectation was met. For a "must not" expectation, that means the agent did not do it.
- fail: what you see shows the expectation was broken.
- cannot_tell: what you see is not enough to decide. Choose it whenever the answer depends on something you cannot see, for example whether the agent really called a tool.
- A message marked (not sent to the agent) was written after the conversation had ended. The agent never saw it, so it cannot be blamed for not answering it.
- Give your answer by calling the tool report_verdict once. After it accepts your verdict, reply with the single word Done."""

_VIEW_TRANSCRIPT = """What you can see: the conversation only. You cannot see the agent's own instruction, the tools it can use, or what those tools returned.

Some expectations allow the agent to state something only if it was given that information. You cannot check what it was given, so use this rule: facts about this customer's own case (the status of an order, an amount, the items) may come from the agent's tools. A general rule, a policy, a time span or a payout method is not such a fact. If the agent states one, treat it as not given to the agent."""

_VIEW_TOOL_LOG = """What you can see: the conversation, and a numbered list of the tool calls the agent really made while it talked, in the order it made them, each with what the tool returned. You cannot see the agent's own instruction. The list does not say which message a call belongs to. The list shows what the agent really did: where the agent's words and the list disagree, the list is right.

Some expectations allow the agent to state something only if it was given that information. Anything a tool returned was given to the agent. A rule, a policy, a time span or a payout method that appears in no tool result is treated as not given."""


def _system_instruction(tool_log_given: bool) -> str:
    return f"{_RULES}\n\n{_VIEW_TOOL_LOG if tool_log_given else _VIEW_TRANSCRIPT}"


# The words _RULES uses for a message the agent never saw.
_NOT_SENT = "(not sent to the agent)"

_SPEAKER_NAMES = {"customer": "Customer", "agent": "Agent"}


def _format_transcript(transcript: list[dict[str, Any]]) -> str:
    """The conversation as one block per message, each under a header such as `Message 2, Agent:`.

    A header is used, and not a line that starts with `2.`, because the agent's
    replies often hold numbered lists of their own, which would look like messages.

    In a finished run, a customer message at the very end is the goodbye that the
    customer wrote after it called `finish`. It was never sent to the agent, so
    its header says so.

    Args:
        transcript: The `transcript` of a run in the results file: dicts with
            `number`, `speaker` (`customer` or `agent`) and `text`.

    Returns:
        The text, with a blank line between messages.
    """
    if not transcript:
        return "(the conversation is empty)"
    last = len(transcript) - 1
    blocks = []
    for index, message in enumerate(transcript):
        speaker = _SPEAKER_NAMES.get(message["speaker"], str(message["speaker"]).capitalize())
        unsent = f" {_NOT_SENT}" if index == last and message["speaker"] == "customer" else ""
        blocks.append(f"Message {message['number']}, {speaker}{unsent}:\n{message['text']}")
    return "\n\n".join(blocks)


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """The tool calls as one numbered line each, oldest first.

    Args:
        tool_calls: The `tool_calls` of a run in the results file: dicts with
            `name`, `arguments`, `result` and `error`.

    Returns:
        Lines such as `Tool call 1: get_order({"order_id": "O1"}) returned {"ok": true}`.
        An empty list is said in words: the target keeps a log and nothing was called.
    """
    if not tool_calls:
        return "(none: the agent called no tools)"
    lines = []
    for number, call in enumerate(tool_calls, start=1):
        arguments = json.dumps(call["arguments"], ensure_ascii=False)
        result = json.dumps(call["result"], ensure_ascii=False)
        line = f"Tool call {number}: {call['name']}({arguments}) returned {result}"
        if call.get("error"):
            line += f" (error: {call['error']})"
        lines.append(line)
    return "\n".join(lines)


def _build_user_message(
    expectation: str,
    transcript: list[dict[str, Any]],
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> str:
    """The first message of a judgement: the expectation, the conversation and, if there is one, the tool log.

    Args:
        expectation: The sentence to judge.
        transcript: The `transcript` of the run, see `_format_transcript`.
        tool_calls: The `tool_calls` of the run. None means the model gets no
            log, which is not the same as an empty list: that says no tools were called.

    Returns:
        The text to send to the model.
    """
    parts = [
        f"Expectation to judge:\n{expectation}",
        f"Conversation:\n\n{_format_transcript(transcript)}",
    ]
    if tool_calls is not None:
        parts.append(f"Tool calls the agent made:\n{_format_tool_calls(tool_calls)}")
    return "\n\n".join(parts)


@dataclass
class Judgement:
    """What one judgement left behind, whether the judge reached a verdict or not.

    Attributes:
        scenario: Name of the scenario the transcript belongs to.
        expectation_number: Position of the expectation in the scenario, from 1.
        expectation: The sentence that was judged.
        usage: The model calls and tokens of this judgement alone.
        tool_log_given: Whether the model was given the target's tool log.
        verdict: What the judge decided. None if it reported none.
        evidence: The passages the verdict rests on.
        reasoning: How the evidence leads to the verdict.
        error_message: What went wrong, for a person to read. It can be set even
            when there is a verdict: the failure then came after the model had
            reported, e.g. in its closing "Done." reply, and the verdict stands.
        error_type: The class name of the error, e.g. `ModelError`.
    """

    scenario: str
    expectation_number: int
    expectation: str
    usage: Usage
    tool_log_given: bool
    verdict: Optional[Verdict] = None
    evidence: Optional[str] = None
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def failed(self) -> bool:
        """True if the judge reached no verdict.

        An `error_message` alone does not make a judgement failed: if the model
        had reported before the error came, the verdict stands.
        """
        return self.verdict is None

    def to_dict(self) -> dict[str, Any]:
        """The judgement as a plain dict, ready for JSON."""
        return {
            "scenario": self.scenario,
            "expectation_number": self.expectation_number,
            "expectation": self.expectation,
            "verdict": self.verdict.value if self.verdict is not None else None,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "usage": self.usage.to_dict(),
            "tool_log_given": self.tool_log_given,
        }


class Judge:
    """An AI agent that judges one expectation of one finished run.

    It reads the expectation and the transcript, and the target's tool log if
    there is one, and ends by calling the `report_verdict` tool, which sets
    `verdict`, `evidence` and `reasoning`. It never sees the persona, the goal,
    the customer's outcome or the other expectations.

    A Judge keeps what the model reports, so it serves ONE judgement. Make a new
    one for each expectation, the way `run_scenario` makes a new customer for
    each run. Once it has a verdict, `judge()` refuses to run again, because the
    old verdict would be returned as if it were new.

    Attributes:
        scenario: Name of the scenario the transcript belongs to.
        expectation_number: Position of the expectation in the scenario, from 1.
        expectation: The sentence to judge.
        tool_log_given: Whether the model gets the tool log.
        verdict: What the model decided. None until it reports.
        evidence: The passages the verdict rests on. None until it reports.
        reasoning: How the evidence leads to the verdict. None until it reports.
    """

    def __init__(
        self,
        scenario: str,
        expectation_number: int,
        expectation: str,
        transcript: list[dict[str, Any]],
        tool_calls: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Args:
            scenario: Name of the scenario the transcript belongs to.
            expectation_number: Position of the expectation in the scenario, from 1.
            expectation: The sentence to judge.
            transcript: The `transcript` of the run in the results file.
            tool_calls: The `tool_calls` of the run. None means the model gets no
                log: the run has none, or the log is not used. An empty list is
                a log with no calls in it.
        """
        self.scenario = scenario
        self.expectation_number = expectation_number
        self.expectation = expectation
        self.tool_log_given = tool_calls is not None
        self.verdict: Optional[Verdict] = None
        self.evidence: Optional[str] = None
        self.reasoning: Optional[str] = None
        self._spec = AgentSpec(
            system_instruction=_system_instruction(self.tool_log_given),
            tool_schemas=[REPORT_SCHEMA],
        )
        self._user_message = _build_user_message(expectation, transcript, tool_calls)
        self._tools: Tools = {"report_verdict": self._report_verdict}

    async def _report_verdict(self, evidence: str, reasoning: str, verdict: str) -> dict:
        """The `report_verdict` tool: record the verdict, once.

        Args:
            evidence: The passages the verdict rests on.
            reasoning: How the evidence leads to the verdict.
            verdict: `pass`, `fail` or `cannot_tell`, as the model wrote it.

        Returns:
            {"ok": True} once the verdict is recorded.

        Raises:
            ValueError: If a verdict was already reported, if `verdict` is not one
                of the choices, or if `evidence` or `reasoning` is blank. The agent
                loop reports this to the model, which can try again.
        """
        if self.verdict is not None:
            raise ValueError("A verdict was already reported. Reply with the single word Done.")
        if verdict not in _CHOOSABLE_VERDICTS:
            choices = ", ".join(k.value for k in _CHOOSABLE_VERDICTS)
            raise ValueError(f"Verdict must be one of: {choices}.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("`evidence` should not be empty.")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("`reasoning` should not be empty.")
        self.verdict = Verdict(verdict)
        self.evidence = evidence.strip()
        self.reasoning = reasoning.strip()
        return {"ok": True}

    async def judge(self, timeout: float = TIMEOUT, max_calls: int = MAX_CALLS) -> Judgement:
        """Let the model judge the expectation, and report how it went. Never raises.

        A timeout or any failure ends only this judgement and becomes its
        `error_message` and `error_type`, and the cost is kept. So many judgements
        can go side by side, and one that breaks does not stop the others.

        Only cancellation passes through, so that Ctrl+C still stops the program.

        Args:
            timeout: Seconds the judgement may take.
            max_calls: The most model calls the judgement may make. Cancelled
                calls count too.

        Returns:
            The judgement. If the model reported no verdict, `verdict` is None.

        Raises:
            RuntimeError: If this Judge already has a verdict from an earlier call.
                That is a mistake in the caller, not a failure of the judgement.
        """
        if self.verdict is not None:
            raise RuntimeError("This Judge has already judged. Make a new one for each expectation.")
        limit = asyncio.timeout(timeout)
        history: History = []
        error_message: Optional[str] = None
        error_type: Optional[str] = None

        with track_usage(max_calls) as usage:
            try:
                async with limit:
                    await get_complete_reply(self._tools, history, self._user_message, self._spec)
                if self.verdict is None:
                    raise JudgeError("The judge ended without reporting a verdict.")
            except TimeoutError as e:
                # A TimeoutError can also come from the model client, e.g. a socket. It is
                # our timeout only if the limit says it ran out.
                if limit.expired():
                    error_message = f"The judgement took longer than its timeout of {timeout:g} seconds."
                else:
                    error_message = str(e) or type(e).__name__
                error_type = type(e).__name__
            except Exception as e:
                error_message = str(e) or type(e).__name__
                error_type = type(e).__name__

        return Judgement(
            scenario=self.scenario,
            expectation_number=self.expectation_number,
            expectation=self.expectation,
            usage=usage,
            tool_log_given=self.tool_log_given,
            verdict=self.verdict,
            evidence=self.evidence,
            reasoning=self.reasoning,
            error_message=error_message,
            error_type=error_type,
        )

def _bad_run(name: str, problem: str) -> JudgeInputError:
    """The error for a run of the results that the judge cannot read."""
    return JudgeInputError(f"The run {name!r} in the results {problem}.")

