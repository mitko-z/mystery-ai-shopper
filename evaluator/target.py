from abc import abstractmethod
from typing import Protocol, runtime_checkable

from core.tool_recorder import ToolCall


class TargetError(Exception):
    """Raised when the target could not give a final answer."""


class Target(Protocol):
    @abstractmethod
    async def send(self, message: str) -> str:
        """Send one message to the target and return its reply.

        Args:
            message: What the customer wrote.
        Returns:
            The text of the target's final answer.
        Raises:
            TargetError: Adapters wrap their own failures (HTTP, serial, ...)
            in TargetError
        """
        ...


@runtime_checkable
class ToolCallSource(Protocol):
    """Optional extra for a target that can say which tools its agent called.

    A black-box target has no such log, and nothing may require one. Where a log
    exists (our own stand-in shop, a vendor's trace API), it is ground truth
    that a judge can compare a transcript-only verdict with. Check for it with
    `isinstance(target, ToolCallSource)`.
    """

    def tool_calls(self) -> list[ToolCall]:
        """The tool calls the agent has made so far, oldest first.

        An empty list means the target keeps a log and nothing has been called
        yet. A target that keeps no log does not have this method at all, so
        "no log" is never mistaken for "no calls".
        """
        ...
