import asyncio
import sys

from core import agent_loop
from core.agent_loop import AgentSpec
from core.app_types import Tools
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

SPEC = AgentSpec(
    system_instruction="you are a support agent for this shop; use the tools to answer",
    tool_schemas=[
        get_order_function,
        check_stock_function,
        refund_order_function
    ],
)


def tools_for(shop: store.Shop) -> Tools:
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


if __name__ == "__main__":
    history = []
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:])
    else:
        user_message = "Hi! I want my order O1001 to be fully refunded! I am not satisfied with the product at all."
    print(f"User message: {user_message}")

    from shop import seed
    asyncio.run(agent_loop.get_complete_reply(tools_for(seed.get_seed_shop()), history, user_message, SPEC))

    print("Updated history:", history)
