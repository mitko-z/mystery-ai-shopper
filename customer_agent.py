from google import genai
from google.genai import types
import config

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
            "amount": {"type": "number", "description": "The amount to refund, in cents."}
        },
        "required": ["order_id"]
    }
}


prompt = input("Enter your prompt: ")
interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=prompt,
    system_instruction="you are a support agent for this shop; use the tools to answer",
    tools=[
        get_order_function,
        check_stock_function,
        refund_order_function
    ],
)

tools_to_call = []

for step in interaction.steps:
    if step.type == "function_call":
        tools_to_call.append({
            "name": step.name,
            "arguments": step.arguments
        })

