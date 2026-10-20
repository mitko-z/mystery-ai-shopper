import asyncio
import functools
import json
import logging
from dataclasses import dataclass
from itertools import takewhile
from typing import Any, Optional

from google import genai
from google.genai import types

from core import config
from core.app_types import Tools, Step, History
from core.usage import current_call_limit_reached, current_max_calls, record, record_cancelled

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.6-flash"

MAX_TURNS = 8 # max turns the agent can do before replying with a final answer


@dataclass(frozen=True)
class AgentSpec:
    """What makes one agent different from another: who it is and which tools it
    may ask for.
    """

    system_instruction: str
    tool_schemas: list[dict[str, Any]]
    model: str = DEFAULT_MODEL


class TurnLimitError(Exception):
    """Raised when the agent exceeds the maximum number of turns without producing
    a final answer.
    """
    pass


class CallLimitError(Exception):
    """Raised when a run has made all the model calls it may make."""


@functools.cache
def _get_client() -> genai.Client:
    """Create the Gemini client on first use, so importing this module needs
    no API key.

    Returns:
        The shared client, created once and reused.

    Raises:
        config.MissingAPIKeyError: If the API key is not set.
    """
    # The SDK retries 429 by sleeping for the whole Retry-After header. With a
    # daily quota that is hours, so the call looks hung. Leave 429 out of the
    # retried codes so it raises at once, and bound every request.
    return genai.Client(
        api_key=config.get_api_key(),
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(
                attempts=2,
                http_status_codes=[500, 502, 503, 504],
            ),
        ),
    )

async def run_turn(history: History, user_message: Optional[str], spec: AgentSpec) -> None:
    """Run one conversation turn with the agent and record it in `history`.

    `history` is the only output. The turn appends, in order, the user message
    and every step the agent produced: `thought`, `function_call` and
    `model_output`. The agent's text reply is the `model_output` step(s) at the
    end, and the tools it wants are the `function_call` steps after the last
    `user_input`. The tools are not executed here. Each call to the Gemini API
    is recorded in `core.usage`.

    Args:
        history: The conversation so far, as interaction steps. Appended to in
            place, and left untouched if the turn fails. Pass an empty list to
            start a conversation.
        user_message: The customer's message, e.g. "Where is my order O1001?".
            If None, the turn is a follow-up to the previous turn and uses the last
            tool result as input.
        spec: The agent's system instruction, tool schemas and model.

    Raises:
        CallLimitError: If the run being tracked has made as many model calls as
            it may. Nothing is sent to the model then.
        config.MissingAPIKeyError: If the API key is not set.
        ValueError: If the interaction response contains no steps.
    """
    if current_call_limit_reached():
        raise CallLimitError(f"This run reached its limit of {current_max_calls()} model calls.")

    new_input = []
    if user_message is not None:
        new_input.append({"type": "user_input", "content": [{"type": "text", "text": user_message}]})

    try:
        interaction = await _get_client().aio.interactions.create(
            model=spec.model,
            input=[*history, *new_input],
            system_instruction=spec.system_instruction,
            tools=spec.tool_schemas,
        )
    except asyncio.CancelledError:
        # E.g. a timeout hung up on the call. No response will come, so its tokens
        # are unknown, but it is likely billed. Count it, and let the cancel go on.
        record_cancelled()
        raise

    record(interaction.usage)

    if not interaction.steps:
        raise ValueError("No steps found in the interaction response.")

    new_steps = [
        step.model_dump(mode="json", by_alias=True, exclude_none=True)
        for step in interaction.steps
    ]

    history.extend(new_input)
    history.extend(new_steps)

    tools_to_call = [
        {"name": step["name"], "arguments": step["arguments"]}
        for step in new_steps
        if step["type"] == "function_call"
    ]
    logger.debug("Tools to call: %s", tools_to_call)


async def _execute_call(
    tools: Tools, call: dict[str, Any]
) -> Step:
    """Run the tool a `function_call` step asks for and build its `function_result` step.

    Never raises: an unknown tool or a tool that fails becomes an error result,
    so the agent sees what went wrong and can recover.

    Args:
        tools: A dictionary of available tools.
        call: A `function_call` step, with `id`, `name` and `arguments`.

    Returns:
        The `function_result` step, with `call_id` set to the call's `id`.
    """
    name = call.get("name")
    if name in tools:
        try:
            result = await tools[name](**call["arguments"])
        except Exception as e:
            result = {"ok": False, "error": str(e)}
    else:
        result = {"ok": False, "error": f"Tool {name} not found."}
    return _function_result(call, result)


def _function_result(call: dict[str, Any], result: dict[str, Any]) -> Step:
    """Build the `function_result` step that answers a `function_call` step.

    Args:
        call: The `function_call` step being answered.
        result: The tool's result, `{"ok": True, ...}` or `{"ok": False, "error": ...}`.

    Returns:
        The `function_result` step, with `call_id` set to the call's `id`.
    """
    return {
        "type": "function_result",
        "call_id": call.get("id"),
        "name": call.get("name"),
        "result": json.dumps(result),
        "is_error": not result.get("ok", False),
    }


async def get_complete_reply(
        tools: Tools,
        history: History,
        user_message: str,
        spec: AgentSpec) -> None:
    """Run conversation turns until the agent produces a `model_output` step, or
    until `MAX_TURNS` is reached. If the agent produces a `function_call`,
    execute the tool and feed the result back to the agent in the next turn.

    On the last turn the agent cannot see another result, so tools it asks for
    are not executed: each call is answered with an error result instead.

    Args:
        tools: A dictionary of available tools.
        history: The conversation so far, as interaction steps. Append to it
            in place.
        user_message: The customer's message.
        spec: The agent's system instruction, tool schemas and model.

    Raises:
        TurnLimitError: If the agent exceeds `MAX_TURNS` without producing a
            final answer or if it fails to produce a valid turn.
    """
    for turn in range(MAX_TURNS):
        last_turn = turn == MAX_TURNS - 1
        try:
            await run_turn(history, user_message, spec)
        except ValueError as e:
            raise TurnLimitError(f"Agent failed to produce a valid turn: {e}")

        if history[-1].get("type") == "model_output":
            break
        user_message = None  # Only the first turn uses the user's message; subsequent turns use tool results.

        # A call is pending until a `function_result` with its id is in `history`.
        # Deriving this from `history` keeps calls from earlier conversation turns
        # from running again.
        answered = {s.get("call_id") for s in history if s.get("type") == "function_result"}
        pending = [
            s for s in history
            if s.get("type") == "function_call" and s.get("id") not in answered
        ]
        for call in pending:
            if last_turn:
                # The agent would not get another turn to see the result, and the API
                # cannot forbid tool calls, so do not run the tool. Answering the call
                # keeps `history` valid and tells the agent it did not run.
                result = {"ok": False, "error": "Not executed: the turn limit was reached."}
                history.append(_function_result(call, result))
            else:
                history.append(await _execute_call(tools, call))
    else:
        raise TurnLimitError(f"The agent could not provide a complete answer in {MAX_TURNS} turns. Please contact support.")


def final_text(history: History) -> str:
    """Return the text of the agent's answer to the latest message.

    `get_complete_reply` hands everything back through `history`, thoughts, tool
    calls and tool results included. The answer is the `model_output` step(s)
    at the end; the earlier steps of the turn are not for the customer. One
    model response can end with several such steps, and its text parts are
    joined as they are, the way the SDK builds `Interaction.output_text`.

    Args:
        history: A conversation whose last turn ended with a `model_output` step.

    Returns:
        The reply text, or "" if those steps carry no text.
    """
    tail = takewhile(lambda step: step.get("type") == "model_output", reversed(history))
    return "".join(
        part.get("text", "")
        for step in reversed(list(tail))
        for part in step.get("content", [])
        if part.get("type") == "text"
    )
