from abc import abstractmethod
from typing import Protocol


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

