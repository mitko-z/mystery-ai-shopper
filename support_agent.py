import functools
import json
from typing import Any, Optional
import sys

from google import genai
from google.genai import types
import config
import asyncio

from shop import store


get_order_function = {
    "type": "function",
    "name": "get_order",
    "description": "Retrieves information about a specific order.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The ID of the order to retrieve."}
        },
        "required": ["order_id"]
    }
}

check_stock_function = {
    "type": "function",
    "name": "check_stock",
    "description": "Checks the stock availability of a specific product.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "description": "The ID of the product to check."}
        },
        "required": ["product_id"]
    }
}

refund_order_function = {
    "type": "function",
    "name": "refund_order",
    "description": "Processes a refund for a specific order.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The ID of the order to refund."},
            "amount": {"type": "integer", "description": "The amount to refund, in cents."}
        },
        "required": ["order_id"]
    }
}

MAX_TURNS = 8 # max turns the agent can do before replying with a final answer

class TurnLimitError(Exception):
    """Raised when the agent exceeds the maximum number of turns without producing a final answer."""
    pass

def _tools_for(shop: store.Shop) -> dict[str, Any]:
    """Build the tool definitions for the support agent, using the shop's methods.

    Args:
        shop: The shop instance whose methods are used as tools.

    Returns:
        A dictionary mapping tool names to their corresponding methods.
    """
    return {
        "get_order": shop.get_order,
        "check_stock": shop.check_stock,
        "refund_order": shop.refund_order
    }

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

async def run_turn(history: list[dict[str, Any]], user_message: Optional[str]) -> None:
    """Run one conversation turn with the support agent and record it in `history`.

    `history` is the only output. The turn appends, in order, the user message
    and every step the agent produced: `thought`, `function_call` and
    `model_output`. The agent's text reply is the `model_output` step(s) at the
    end, and the tools it wants are the `function_call` steps after the last
    `user_input`. The tools are not executed here.

    Args:
        history: The conversation so far, as interaction steps. Appended to in
            place, and left untouched if the turn fails. Pass an empty list to
            start a conversation.
        user_message: The customer's message, e.g. "Where is my order O1001?".
            If None, the turn is a follow-up to the previous turn and uses the last
            tool result as input.

    Raises:
        config.MissingAPIKeyError: If the API key is not set.
        ValueError: If the interaction response contains no steps.
    """
    new_input = []
    if user_message is not None:
        new_input.append({"type": "user_input", "content": [{"type": "text", "text": user_message}]})
    interaction = await _get_client().aio.interactions.create(
        model="gemini-3.6-flash",
        input=[*history, *new_input],
        system_instruction="you are a support agent for this shop; use the tools to answer",
        tools=[
            get_order_function,
            check_stock_function,
            refund_order_function
        ],
    )

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
    print(f"Tools to call: {tools_to_call}")


async def _execute_call(shop: store.Shop, call: dict[str, Any]) -> dict[str, Any]:
    """Run the tool a `function_call` step asks for and build its `function_result` step.

    Never raises: an unknown tool or a tool that fails becomes an error result,
    so the agent sees what went wrong and can recover.

    Args:
        shop: The shop instance to use for tool calls.
        call: A `function_call` step, with `id`, `name` and `arguments`.

    Returns:
        The `function_result` step, with `call_id` set to the call's `id`.
    """
    name = call.get("name")
    tools = _tools_for(shop)
    if name in tools:
        try:
            result = await tools[name](**call["arguments"])
        except Exception as e:
            result = {"ok": False, "error": str(e)}
    else:
        result = {"ok": False, "error": f"Tool {name} not found."}
    return _function_result(call, result)


def _function_result(call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
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


async def get_complete_reply(shop: store.Shop, history: list[dict[str, Any]], user_message: str) -> None:
    """Run conversation turns until the agent produces a `model_output` step, or
    until `MAX_TURNS` is reached. If the agent produces a `function_call`,
    execute the tool and feed the result back to the agent in the next turn.

    On the last turn the agent cannot see another result, so tools it asks for
    are not executed: each call is answered with an error result instead.

    Args:
        shop: The shop instance to use for tool calls.
        history: The conversation so far, as interaction steps. Append to it
            in place.
        user_message: The customer's message.

    Raises:
        TurnLimitError: If the agent exceeds `MAX_TURNS` without producing a
            final answer or if it fails to produce a valid turn.
    """
    for turn in range(MAX_TURNS):
        last_turn = turn == MAX_TURNS - 1
        try:
            await run_turn(history, user_message)
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
                history.append(await _execute_call(shop, call))
    else:
        raise TurnLimitError(f"The agent could not provide a complete answer in {MAX_TURNS} turns. Please contact support.")

if __name__ == "__main__":
    history = []
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:])
    else:
        user_message = "Hi! I want my order O1001 to be fully refunded! I am not satisfied with the product at all."
    print(f"User message: {user_message}")

    from shop import seed
    asyncio.run(get_complete_reply(seed.get_seed_shop(), history, user_message))

    print("Updated history:", history)
