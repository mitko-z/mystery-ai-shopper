import asyncio
from dataclasses import dataclass, field

from core import agent_loop
from core.app_types import History
from core.tool_recorder import ToolCall, recording
from shop import seed, support_agent
from shop.store import Shop


class EmptyReplyError(RuntimeError):
    """Raised when the agent's final answer has no text."""


@dataclass
class SupportSession:
    """One conversation with the support agent, in a shop of its own.

    The HTTP app keeps one per `conversation_id`; a test run can use one directly.
    """

    history: History = field(default_factory=list)
    shop: Shop = field(default_factory=seed.get_seed_shop)
    tool_calls: list[ToolCall] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, message: str) -> str:
        """Send one customer message to the support agent and return its reply.

        Messages of one session are handled one at a time, in arrival order.
        Every tool the agent calls is appended to `tool_calls`. After a
        failure the session is kept, so the customer can write again.

        Args:
            message: What the customer wrote.

        Returns:
            The text of the agent's final answer.

        Raises:
            config.MissingAPIKeyError: If the API key is not set.
            agent_loop.TurnLimitError: If the agent could not give a final answer.
            EmptyReplyError: If the final answer has no text.
        """
        async with self.lock:
            tools = recording(support_agent.tools_for(self.shop), self.tool_calls)
            await agent_loop.get_complete_reply(tools, self.history, message, support_agent.SPEC)
            reply = agent_loop.final_text(self.history)
            if not reply.strip():
                raise EmptyReplyError("The final answer has no text.")
            return reply
