import functools
from typing import Any

from google import genai
import config


@functools.cache
def _get_client() -> genai.Client:
    """Create the Gemini client on first use, so importing this module needs no API key.

    Returns:
        The shared client, created once and reused.

    Raises:
        config.MissingAPIKeyError: If the API key is not set.
    """
    return genai.Client(api_key=config.get_api_key())


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


async def run_turn(history: list[dict[str, Any]], user_message: str) -> None:
    """Run one conversation turn with the support agent and record it in `history`.

    `history` is the only output. The turn appends, in order, the user message
    and every step the agent produced: `thought`, `function_call` and
    `model_output`. The agent's text reply is the `model_output` step(s) at the
    end, and the tools it wants are the `function_call` steps after the last
    `user_input`. The tools are not executed here. Once they are, append a
    `function_result` step for each call (matching its `id` as `call_id`)
    before the next turn.

    Args:
        history: The conversation so far, as interaction steps. Appended to in
            place, and left untouched if the turn fails. Pass an empty list to
            start a conversation.
        user_message: The customer's message, e.g. "Where is my order O1001?".

    Raises:
        config.MissingAPIKeyError: If the API key is not set.
        ValueError: If the interaction response contains no steps.
    """
    user_step = {"type": "user_input", "content": [{"type": "text", "text": user_message}]}
    interaction = await _get_client().aio.interactions.create(
        model="gemini-3.5-flash",
        input=[*history, user_step],
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

    history.append(user_step)
    history.extend(new_steps)

    tools_to_call = [
        {"name": step["name"], "arguments": step["arguments"]}
        for step in new_steps
        if step["type"] == "function_call"
    ]
    print(f"Tools to call: {tools_to_call}")