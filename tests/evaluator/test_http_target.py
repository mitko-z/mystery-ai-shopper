import json

import httpx
import pytest

from evaluator.http_target import HttpChatTarget
from evaluator.target import TargetError

pytestmark = pytest.mark.anyio


@pytest.fixture
async def make_target():
    """Build a target whose HTTP client is answered by `handler`; the clients are closed after the test."""
    clients = []

    def make(handler, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
        clients.append(client)
        return HttpChatTarget(client, "conv-1", **kwargs)

    yield make
    for client in clients:
        await client.aclose()


async def test_message_is_posted_and_the_reply_returned(make_target):
    """The id and the message go to /chat as JSON, and the reply text comes back."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"conversation_id": "conv-1", "reply": "Hello!"})

    reply = await make_target(handler).send("hi")

    assert reply == "Hello!"
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/chat"
    assert json.loads(requests[0].content) == {"conversation_id": "conv-1", "message": "hi"}


async def test_the_endpoint_path_can_be_chosen(make_target):
    """A system whose chat endpoint is not /chat can still be reached."""
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"reply": "ok"})

    await make_target(handler, path="/api/support").send("hi")

    assert paths == ["/api/support"]


async def test_an_error_status_is_a_target_error(make_target):
    """The status and the start of the body are in the error, so a failed run can be understood."""
    target = make_target(lambda request: httpx.Response(502, json={"detail": "agent unavailable"}))

    with pytest.raises(TargetError, match="502.*agent unavailable"):
        await target.send("hi")


async def test_an_unreachable_target_is_a_target_error(make_target):
    """A connection failure is reported as a target error, not as an httpx one."""
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TargetError, match="could not be reached"):
        await make_target(handler).send("hi")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"answer": "no reply field"}),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"reply": 42}),
        httpx.Response(200, text="not json at all"),
    ],
    ids=["no-reply-field", "not-an-object", "reply-not-text", "not-json"],
)
async def test_an_answer_without_reply_text_is_a_target_error(make_target, response):
    """Whatever is wrong with the body, the runner only ever sees a TargetError."""
    with pytest.raises(TargetError):
        await make_target(lambda request: response).send("hi")
