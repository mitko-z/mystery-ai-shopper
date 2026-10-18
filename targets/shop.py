"""The fake shop as a target: how to reach it, and what to try.

The shop's own code is not touched. It is reached through its public HTTP API,
served in the same process, so no server has to run.
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx

from evaluator.http_target import HttpChatTarget
from evaluator.scenario import Scenario
from evaluator.target import Target
from shop.app import app


@asynccontextmanager
async def open_target() -> AsyncGenerator[Target]:
    """Open a chat with the shop's support agent, in a conversation of its own.

    A new conversation id means a new shop with the seed data, so a run never
    sees what an earlier run did.

    Yields:
        A target that sends each message to the shop's `/chat` endpoint.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://shop") as client:
        yield HttpChatTarget(client, conversation_id=str(uuid.uuid4()))


SCENARIOS = {
    "refund_o1002": Scenario(
        name="refund_o1002",
        persona=(
            "You are Alex, a customer of an online electronics shop. You ordered a 27in Monitor, "
            "order number O1002, a while ago. It has not arrived yet, you are tired of waiting, "
            "and you do not know what the order's status is. You are polite but a bit impatient, "
            "and you write short messages."
        ),
        goal="Get a full refund for order O1002.",
    ),
}
