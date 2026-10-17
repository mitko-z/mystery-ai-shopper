"""A module for custom types."""


from collections.abc import Awaitable, Callable, Mapping
from typing import Any

Tools = Mapping[str, Callable[..., Awaitable[Any]]]
"""A type alias for a dictionary of tools, where each tool is a
callable that returns an awaitable result."""

Step = dict[str, Any]
"""A type alias for a step in the conversation history, represented
as a dictionary."""

History = list[Step]
"""A type alias for the conversation history, which is a list of steps."""
