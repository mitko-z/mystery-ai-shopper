from collections.abc import Mapping, Sequence
from typing import Any, Optional

import httpx

from evaluator.target import Target, TargetError

# The markers of a request body template. A value that is exactly a marker is
# replaced by the message or the conversation id. Anything else is sent as written.
MESSAGE_MARKER = "{message}"
CONVERSATION_ID_MARKER = "{conversation_id}"
MARKERS = (MESSAGE_MARKER, CONVERSATION_ID_MARKER)

DEFAULT_BODY_TEMPLATE: Mapping[str, Any] = {
    "conversation_id": CONVERSATION_ID_MARKER,
    "message": MESSAGE_MARKER,
}
DEFAULT_REPLY_PATH: tuple[str, ...] = ("reply",)


def fill_template(template: Any, message: str, conversation_id: str) -> Any:
    """A copy of `template` with each marker value replaced.

    Only a value that is exactly a marker is replaced, never a part of a text,
    so a message that holds braces or a marker's name is sent as it is.

    Args:
        template: A JSON-like value: dicts, lists, text, numbers, booleans.
        message: What replaces `{message}`.
        conversation_id: What replaces `{conversation_id}`.

    Returns:
        The filled-in copy. The template itself is not changed.
    """
    if isinstance(template, Mapping):
        return {key: fill_template(value, message, conversation_id) for key, value in template.items()}
    if isinstance(template, list):
        return [fill_template(item, message, conversation_id) for item in template]
    if template == MESSAGE_MARKER:
        return message
    if template == CONVERSATION_ID_MARKER:
        return conversation_id
    return template


def read_path(data: Any, path: Sequence[str]) -> Any:
    """The value at `path` in a decoded JSON answer.

    A step is a key of an object, or the index of a list item when the value
    there is a list, e.g. `("choices", "0", "text")`.

    Raises:
        KeyError: If a step does not lead anywhere.
    """
    for step in path:
        if isinstance(data, Mapping) and step in data:
            data = data[step]
        elif isinstance(data, list) and step.isdigit() and int(step) < len(data):
            data = data[int(step)]
        else:
            raise KeyError(step)
    return data


class HttpChatTarget(Target):
    """A target that is reached over HTTP with `POST <path>`.

    By default the request body is `{"conversation_id": ..., "message": ...}`
    and the answer is a JSON object with the agent's text in `reply`. Another
    body shape and another place of the reply can be given. The client decides
    where the requests go (a live URL, or an ASGI app in the same process) and
    which headers they carry.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        conversation_id: str,
        path: str = "/chat",
        body_template: Optional[Mapping[str, Any]] = None,
        reply_path: Sequence[str] = DEFAULT_REPLY_PATH,
    ) -> None:
        """
        Args:
            client: The client to send requests with. The caller closes it.
            conversation_id: Id that keeps all messages of one run in one conversation.
            path: The chat endpoint.
            body_template: The request body, with `{message}` and `{conversation_id}`
                as whole values where they go (see `fill_template`). None means
                the default body.
            reply_path: Where the agent's text is in the JSON answer, one step
                per key or list index (see `read_path`).
        """
        self._client = client
        self._conversation_id = conversation_id
        self._path = path
        self._body_template = DEFAULT_BODY_TEMPLATE if body_template is None else body_template
        self._reply_path = tuple(reply_path)

    @property
    def conversation_id(self) -> str:
        """The id that keeps all messages of this run in one conversation."""
        return self._conversation_id

    async def send(self, message: str) -> str:
        """Send one message to the chat endpoint and return the agent's reply.

        Args:
            message: What the customer wrote.

        Returns:
            The text at the reply path of the response.

        Raises:
            TargetError: If the endpoint cannot be reached, answers with an
                error status, or the answer has no text at the reply path.
        """
        try:
            response = await self._client.post(
                self._path,
                json=fill_template(self._body_template, message, self._conversation_id),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TargetError(
                f"The target answered {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.HTTPError as e:
            raise TargetError(f"The target could not be reached: {e!r}") from e

        where = ".".join(self._reply_path)
        try:
            reply = read_path(response.json(), self._reply_path)
        except (ValueError, KeyError) as e:
            raise TargetError(f"The target's answer has no `{where}` field.") from e
        if not isinstance(reply, str):
            raise TargetError(f"The target's `{where}` is not text.")
        return reply
