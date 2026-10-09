from typing import Any, TypedDict

from google import genai
import config


class AgentResponse(TypedDict):
    """What the support agent produced for one prompt."""

    text: str
    tool_calls: list[dict[str, Any]]


client = genai.Client(api_key=config.get_api_key())


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


async def run_support_agent(prompt: str) -> AgentResponse:
    """Send a customer prompt to the support agent and return its reply.

    The tools are not executed here; only the requested calls are returned.

    Args:
        prompt: The customer's message, e.g. "Where is my order O1001?".

    Returns:
        {"text": str, "tool_calls": list}. `text` is the agent's text reply
        ("" if it produced none). `tool_calls` holds one {"name": str,
        "arguments": dict} per `function_call` step, and is empty if the
        agent replied without requesting any tool.

    Raises:
        ValueError: If the interaction response contains no steps.
    """
    interaction = await client.aio.interactions.create(
        model="gemini-3.5-flash",
        input=prompt,
        system_instruction="you are a support agent for this shop; use the tools to answer",
        tools=[
            get_order_function,
            check_stock_function,
            refund_order_function
        ],
    )

    if not interaction.steps:
        raise ValueError("No steps found in the interaction response.")

    tools_to_call = []

    for step in interaction.steps:
        if step.type == "function_call":
            tools_to_call.append({
                "name": step.name,
                "arguments": step.arguments
            })

    print(f"Tools to call: {tools_to_call}")
    return {"text": interaction.output_text or "", "tool_calls": tools_to_call}