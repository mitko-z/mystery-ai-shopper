import functools
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from core.app_types import Tools


def _as_text(value: Any) -> str:
    """The text form of a value JSON cannot hold."""
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _json_safe(value: Any) -> Any:
    """Return a copy of `value` that holds only JSON types.

    What JSON cannot hold is replaced by its text. If the value cannot be
    converted piece by piece (a dict with tuple keys, a value that contains
    itself), all of it becomes text. This never raises, because it is used when
    results are written, and one odd tool result must not lose the others.
    """
    try:
        return json.loads(json.dumps(value, default=_as_text))
    except (TypeError, ValueError):
        return _as_text(value)


@dataclass(frozen=True)
class ToolCall:
    """One call the agent made to a tool."""

    name: str
    arguments: dict[str, Any]
    result: Any
    started_at: datetime
    duration: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """The call as a plain dict, ready for JSON.

        The start time is ISO text. The arguments and the result, which can be
        anything, are copied with what JSON cannot hold turned into text.
        """
        return {
            "name": self.name,
            "arguments": _json_safe(self.arguments),
            "result": _json_safe(self.result),
            "started_at": self.started_at.isoformat(),
            "duration": self.duration,
            "error": self.error,
        }


def recording(tools: Tools, log: list[ToolCall]) -> Tools:
    """Return `tools` with every call appended to `log`, then passed through."""

    def wrap(name, fn):
        @functools.wraps(fn)
        async def wrapper(**arguments):
            started_at = datetime.now(timezone.utc)
            t0 = time.perf_counter()
            try:
                result = await fn(**arguments)
            except BaseException as e:
                # BaseException, not Exception: asyncio.wait_for cancels a slow call by
                # raising CancelledError inside it, and that is not an Exception.
                # CancelledError has no message, so fall back to the class name.
                log.append(ToolCall(name, dict(arguments), None, started_at,
                                    time.perf_counter() - t0, error=str(e) or type(e).__name__))
                raise
            log.append(ToolCall(name, dict(arguments), result, started_at,
                                time.perf_counter() - t0))
            return result

        return wrapper

    return {name: wrap(name, fn) for name, fn in tools.items()}
