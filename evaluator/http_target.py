import httpx

from evaluator.target import Target, TargetError


class HttpChatTarget(Target):
    """A target that is reached over HTTP with `POST <path>`.

    The request body is `{"conversation_id": ..., "message": ...}` and the
    answer is a JSON object with the agent's text in `reply`. The client decides
    where the requests go: a live URL, or an ASGI app in the same process.
    """

    def __init__(self, client: httpx.AsyncClient, conversation_id: str, path: str = "/chat") -> None:
        """
        Args:
            client: The client to send requests with. The caller closes it.
            conversation_id: Id that keeps all messages of one run in one conversation.
            path: The chat endpoint.
        """
        self._client = client
        self._conversation_id = conversation_id
        self._path = path

    async def send(self, message: str) -> str:
        """Send one message to the chat endpoint and return the agent's reply.

        Args:
            message: What the customer wrote.

        Returns:
            The `reply` text of the response.

        Raises:
            TargetError: If the endpoint cannot be reached, answers with an
                error status, or the answer has no `reply` text.
        """
        try:
            response = await self._client.post(
                self._path,
                json={"conversation_id": self._conversation_id, "message": message},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TargetError(
                f"The target answered {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.HTTPError as e:
            raise TargetError(f"The target could not be reached: {e!r}") from e

        try:
            reply = response.json()["reply"]
        except (ValueError, KeyError, TypeError) as e:
            raise TargetError("The target's answer has no `reply` field.") from e
        if not isinstance(reply, str):
            raise TargetError("The target's `reply` is not text.")
        return reply
