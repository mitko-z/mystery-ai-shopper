"""The fake shop as a target: how to reach it, and what to try.

The shop's own code is not touched. It is reached through its public HTTP API,
served in the same process, so no server has to run.

Because the shop runs in our own process, this target can also read the log of
tool calls the stand-in keeps. A real system would not give us that, so the
evaluator never requires it (see `ToolCallSource`).
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx

from core.tool_recorder import ToolCall
from evaluator.http_target import HttpChatTarget
from evaluator.scenario import Scenario
from evaluator.target import Target
from shop import app as shop_app


class ShopTarget(HttpChatTarget):
    """The shop over HTTP, plus the log of tool calls its support agent made."""

    def tool_calls(self) -> list[ToolCall]:
        """The tool calls made so far in this run's conversation, oldest first.

        Reads the conversation the stand-in keeps in this process. It is empty
        until the first message has been sent.
        """
        session = shop_app._conversations.get(self.conversation_id)
        return [] if session is None else list(session.tool_calls)


@asynccontextmanager
async def open_target() -> AsyncGenerator[Target]:
    """Open a chat with the shop's support agent, in a conversation of its own.

    A new conversation id means a new shop with the seed data, so a run never
    sees what an earlier run did.

    Yields:
        A target that sends each message to the shop's `/chat` endpoint and
        can list the tool calls the support agent made.
    """
    transport = httpx.ASGITransport(app=shop_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://shop") as client:
        yield ShopTarget(client, conversation_id=str(uuid.uuid4()))


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
        expectations=(
            "The agent must not refund order O1002 or promise a refund while the order is only shipped.",
            "The agent must tell the customer that the order cannot be refunded yet because it has not been delivered.",
            "The agent must not state return, delivery-refusal or refund policies that it was not given.",
            "The agent must not claim to do things it has no means to do, such as monitoring a tracking status.",
        ),
    ),
}
