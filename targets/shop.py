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
from pathlib import Path

import httpx

from core.tool_recorder import ToolCall
from evaluator.http_target import HttpChatTarget
from evaluator.scenario_file import load_scenarios
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


# What to try on the shop. Edit the file to add or change scenarios.
SCENARIO_FILE = Path(__file__).resolve().parent.parent / "scenarios" / "shop.toml"
SCENARIOS = load_scenarios(SCENARIO_FILE)
