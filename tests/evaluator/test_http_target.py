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


async def test_the_conversation_id_can_be_read_back(make_target):
    """The id is public, so a target that adds a side channel can find its conversation."""
    target = make_target(lambda request: httpx.Response(200, json={"reply": "ok"}))

    assert target.conversation_id == "conv-1"


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


# --- another body shape and another place of the reply ---

async def test_the_markers_of_a_body_template_are_filled_in_at_any_depth(make_target):
    """Nested tables and lists are walked; values that are not markers are sent as written."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    template = {
        "thread": {"id": "{conversation_id}"},
        "input": [{"role": "user", "content": "{message}"}],
        "stream": False,
        "max_tokens": 200,
    }
    await make_target(handler, body_template=template).send("Where is my order?")

    assert bodies == [{
        "thread": {"id": "conv-1"},
        "input": [{"role": "user", "content": "Where is my order?"}],
        "stream": False,
        "max_tokens": 200,
    }]


async def test_a_message_that_looks_like_a_marker_is_sent_as_it_is(make_target):
    """Only a template value is replaced, never the customer's text, so no message can change the id."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    await make_target(handler).send("{conversation_id} and {message}")

    assert bodies == [{"conversation_id": "conv-1", "message": "{conversation_id} and {message}"}]


async def test_a_marker_inside_a_longer_text_is_not_replaced(make_target):
    """Only a whole value is a marker. The config loader refuses the other kind, so it never gets here."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    template = {"text": "{message}", "note": "Customer says: {message}", "id": "shopper-{conversation_id}"}
    await make_target(handler, body_template=template).send("hi")

    assert bodies == [{"text": "hi", "note": "Customer says: {message}", "id": "shopper-{conversation_id}"}]


async def test_the_body_template_is_the_same_for_every_message(make_target):
    """Filling in makes a copy, so the second message does not see the first one."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok"})

    template = {"input": {"text": "{message}"}}
    target = make_target(handler, body_template=template)
    await target.send("first")
    await target.send("second")

    assert bodies == [{"input": {"text": "first"}}, {"input": {"text": "second"}}]
    assert template == {"input": {"text": "{message}"}}


async def test_the_reply_can_sit_deeper_in_the_answer(make_target):
    """A step is a key, or the index of a list item."""
    answer = {"data": {"messages": [{"text": "first"}, {"text": "Hello!"}]}}
    target = make_target(lambda request: httpx.Response(200, json=answer), reply_path=("data", "messages", "1", "text"))

    assert await target.send("hi") == "Hello!"


@pytest.mark.parametrize(
    "answer",
    [
        {"data": {}},
        {"data": {"messages": []}},
        {"data": {"messages": {"one": "not a list"}}},
        {"data": "text where an object should be"},
    ],
    ids=["key-missing", "index-past-the-end", "index-into-an-object", "key-into-text"],
)
async def test_a_reply_path_that_leads_nowhere_names_the_path(make_target, answer):
    target = make_target(lambda request: httpx.Response(200, json=answer), reply_path=("data", "messages", "0"))

    with pytest.raises(TargetError, match=r"no `data\.messages\.0` field"):
        await target.send("hi")


async def test_a_reply_path_that_leads_to_no_text_names_the_path(make_target):
    target = make_target(lambda request: httpx.Response(200, json={"data": {"text": ["a", "b"]}}),
                         reply_path=("data", "text"))

    with pytest.raises(TargetError, match=r"`data\.text` is not text"):
        await target.send("hi")


async def test_the_default_reply_error_names_the_reply_field(make_target):
    """The message for the default shape stays what it was."""
    target = make_target(lambda request: httpx.Response(200, json={"answer": "x"}))

    with pytest.raises(TargetError, match=r"^The target's answer has no `reply` field\.$"):
        await target.send("hi")
