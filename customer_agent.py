from google import genai
from google.genai import types
import config

client = genai.Client(api_key=config.get_api_key())
prompt = input("Enter your prompt: ")

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="You are a customer assistant. You tell tge users how to" \
        " access the shop's functionality. Currently, the shop tools are:" \
        " shop.store.get_order(order_id), shop.store.check_stock(product_id) and "
        " shop.store.refund_order(order_id, amount). You should always answer " \
        " by pointing these tools to answer the users' questions. So, if the user " \
        " ask you 'What is the status of my order with # O1003', you should answer " \
        " with 'Please call `shop.store.get_order(O1003)`'. Anything else asked by " \
        " the user you should answer with 'I cannot help you with that'.",
        )
    )

print(f"AI: {response.text}")