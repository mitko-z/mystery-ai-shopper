import functools
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.app_types import Tools


@dataclass(frozen=True)
class ToolCall:
    """One call the agent made to a tool."""

    name: str
    arguments: dict[str, Any]
    result: Any
    started_at: datetime
    duration: float
    error: str | None = None


def recording(tools: Tools, log: list[ToolCall]) -> Tools:
    """Return `tools` with every call appended to `log`, then passed through."""

    def wrap(name, fn):
        @functools.wraps(fn)
        async def wrapper(**arguments):
            started_at = datetime.now(timezone.utc)
            t0 = time.perf_counter()
            try:
                result = await fn(**arguments)
            except Exception as e:
                log.append(ToolCall(name, dict(arguments), None, started_at,
                                    time.perf_counter() - t0, error=str(e)))
                raise
            log.append(ToolCall(name, dict(arguments), result, started_at,
                                time.perf_counter() - t0))
            return result

        return wrapper

    return {name: wrap(name, fn) for name, fn in tools.items()}
