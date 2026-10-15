import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

import config
import main
from tests.test_support_agent import (
    FakeInteractions,
    FakeStep,
    ScriptedInteractions,
    call_step,
    reply_step,
    use_fake,
    user_step,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
def fresh_state():
    """Forget the stored conversations before every test. A new one gets its own shop."""
    main._conversations.clear()


@pytest.fixture
async def client():
    """An HTTP client that talks to the app in-process, on the test's event loop."""
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def use_model(monkeypatch, fake):
    """Make the app's agent talk to `fake`.

    The fake client only has `.aio`, so any use of the sync client fails the test.
    """
    use_fake(main.support_agent, monkeypatch, fake)


def shop_of(conversation_id):
    """The shop that belongs to a conversation the app has started."""
    return main._conversations[conversation_id].shop


def say(client, conversation_id, message):
    """POST one customer message to /chat."""
    return client.post("/chat", json={"conversation_id": conversation_id, "message": message})


class FailingInteractions:
    """Stands in for `client.aio.interactions`; every call raises `error`."""

    def __init__(self, error):
        self.calls = []
        self._error = error

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self._error


class GatedInteractions:
    """Stands in for `client.aio.interactions`; the first call waits until `release` is set."""

    def __init__(self, *replies):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._replies = replies

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        if n == 1:
            self.started.set()
            await self.release.wait()
        return SimpleNamespace(steps=[FakeStep(**reply_step(self._replies[n - 1]))])


# --- the conversation ---

async def test_three_turn_conversation_ends_in_a_refund(client, monkeypatch):
    """The customer gives the order number on turn 2, and the agent refunds on turn 3."""
    fake = ScriptedInteractions(
        [reply_step("Which order is it about?")],
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [reply_step("O1001 was delivered, 50.00 total. Refund it?")],
        [call_step("c2", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded 50.00.")],
    )
    use_model(monkeypatch, fake)

    first = await say(client, "demo", "I want a refund")
    second = await say(client, "demo", "It is O1001")
    assert (await shop_of("demo").get_order("O1001"))["order"]["refunds"] == []
    third = await say(client, "demo", "Yes, refund it")

    assert [r.status_code for r in (first, second, third)] == [200, 200, 200]
    assert first.json() == {"conversation_id": "demo", "reply": "Which order is it about?"}
    assert second.json()["reply"] == "O1001 was delivered, 50.00 total. Refund it?"
    assert third.json()["reply"] == "Refunded 50.00."
    assert (await shop_of("demo").get_order("O1001"))["order"]["refunds"] == [{"amount": 5000}]
    # The last model call saw the whole conversation, not only the last message.
    sent_users = [s for s in fake.calls[-1]["input"] if s["type"] == "user_input"]
    assert sent_users == [
        user_step("I want a refund"),
        user_step("It is O1001"),
        user_step("Yes, refund it"),
    ]


async def test_conversations_do_not_share_history(client, monkeypatch):
    """A message in one conversation is never sent to the model in another."""
    fake = ScriptedInteractions([reply_step("Hi A")], [reply_step("Hi B")])
    use_model(monkeypatch, fake)

    await say(client, "a", "I am A")
    await say(client, "b", "I am B")

    assert fake.calls[1]["input"] == [user_step("I am B")]


async def test_conversations_do_not_share_a_shop(client, monkeypatch):
    """A refund in one conversation is not seen in another conversation's shop."""
    fake = ScriptedInteractions(
        [call_step("c1", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded 50.00.")],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [reply_step("Nothing refunded on O1001.")],
    )
    use_model(monkeypatch, fake)

    await say(client, "a", "refund O1001")
    await say(client, "b", "what is the status of O1001?")

    assert main._conversations["a"].shop is not main._conversations["b"].shop
    assert (await shop_of("a").get_order("O1001"))["order"]["refunds"] == [{"amount": 5000}]
    assert (await shop_of("b").get_order("O1001"))["order"]["refunds"] == []
    # What the agent of conversation b was told by its own tool call.
    seen_by_b = json.loads(fake.calls[3]["input"][-1]["result"])
    assert seen_by_b["order"]["refunded"] == 0


async def test_one_conversation_keeps_its_shop_between_messages(client, monkeypatch):
    """A refund made on one message is still there on the next message of that conversation."""
    fake = ScriptedInteractions(
        [call_step("c1", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded 50.00.")],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [reply_step("It is fully refunded.")],
    )
    use_model(monkeypatch, fake)

    await say(client, "demo", "refund O1001")
    await say(client, "demo", "is O1001 refunded?")

    seen = json.loads(fake.calls[3]["input"][-1]["result"])
    assert seen["order"]["refunded"] == 5000


async def test_reply_is_the_final_text_only(client, monkeypatch):
    """Thoughts, tool calls and tool results of the turn are not returned."""
    fake = ScriptedInteractions(
        [{"type": "thought", "signature": "sig"}, call_step("c1", "check_stock", {"product_id": "P1"})],
        [reply_step("In stock. "), reply_step("12 left.")],
    )
    use_model(monkeypatch, fake)

    response = await say(client, "demo", "stock of P1?")

    assert response.json() == {"conversation_id": "demo", "reply": "In stock. 12 left."}


async def test_same_conversation_is_handled_one_message_at_a_time(client, monkeypatch):
    """A second message waits for the first, so the model sees the first one's reply."""
    fake = GatedInteractions("first", "second")
    use_model(monkeypatch, fake)

    one = asyncio.create_task(say(client, "demo", "one"))
    await fake.started.wait()
    two = asyncio.create_task(say(client, "demo", "two"))
    await asyncio.sleep(0.05)
    assert len(fake.calls) == 1  # "two" has not reached the model yet

    fake.release.set()
    first, second = await asyncio.gather(one, two)

    assert (first.json()["reply"], second.json()["reply"]) == ("first", "second")
    assert fake.calls[1]["input"] == [user_step("one"), reply_step("first"), user_step("two")]


# --- bad requests ---

@pytest.mark.parametrize(
    "body",
    [
        {"conversation_id": "demo"},
        {"message": "hi"},
        {"conversation_id": "demo", "message": ""},
        {"conversation_id": "demo", "message": "   "},
        {"conversation_id": "", "message": "hi"},
        {"conversation_id": "x" * 129, "message": "hi"},
        {"conversation_id": "demo", "message": "x" * 4001},
    ],
    ids=["no-message", "no-id", "empty-message", "blank-message", "empty-id", "long-id", "long-message"],
)
async def test_invalid_request_is_rejected(client, monkeypatch, body):
    """A body that is not a usable id plus message gets a 422 and never reaches the model."""
    fake = ScriptedInteractions()
    use_model(monkeypatch, fake)

    response = await client.post("/chat", json=body)

    assert response.status_code == 422
    assert fake.calls == []
    assert main._conversations == {}


# --- failures ---

async def test_missing_api_key_is_a_503(client, monkeypatch):
    """The customer is told the agent is not configured, without the setup details."""
    def no_key():
        raise config.MissingAPIKeyError(f"{config.API_KEY_ENV_VAR} is not set.")

    monkeypatch.setattr(main.support_agent, "_get_client", no_key)

    response = await say(client, "demo", "hi")

    assert response.status_code == 503
    assert config.API_KEY_ENV_VAR not in response.text


async def test_turn_limit_is_a_502_and_the_conversation_goes_on(client, monkeypatch):
    """After the limit error the history is kept, so the next message can continue it."""
    monkeypatch.setattr(main.support_agent, "MAX_TURNS", 2)
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [reply_step("Sorry about that. O1001 is delivered.")],
    )
    use_model(monkeypatch, fake)

    failed = await say(client, "demo", "where is my order?")
    retried = await say(client, "demo", "try again")

    assert failed.status_code == 502
    assert "Please contact support" in failed.json()["detail"]
    assert retried.status_code == 200
    assert retried.json()["reply"] == "Sorry about that. O1001 is delivered."
    sent = fake.calls[2]["input"]
    assert [s["type"] for s in sent] == [
        "user_input", "function_call", "function_result",
        "function_call", "function_result", "user_input",
    ]


@pytest.mark.parametrize(
    "final_step",
    [
        {"type": "model_output"},
        {"type": "model_output", "content": []},
        {"type": "model_output", "content": [{"type": "text", "text": "  "}]},
    ],
    ids=["no-content", "no-parts", "blank-text"],
)
async def test_empty_answer_is_a_502(client, monkeypatch, final_step):
    """A final answer without text is an error, not a 200 with an empty reply."""
    use_model(monkeypatch, ScriptedInteractions([final_step]))

    response = await say(client, "demo", "hi")

    assert response.status_code == 502
    assert "empty answer" in response.json()["detail"]


async def test_response_without_steps_is_a_502(client, monkeypatch):
    """A model response with no steps is reported as a failed turn."""
    use_model(monkeypatch, FakeInteractions(steps=None))

    response = await say(client, "demo", "hi")

    assert response.status_code == 502


@pytest.mark.parametrize(
    "error",
    [
        genai_errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}),
        genai_errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["server-error", "quota", "timeout"],
)
async def test_model_failure_is_a_502_without_details(client, monkeypatch, error):
    """A failing model call becomes a 502 that does not leak the upstream message."""
    fake = FailingInteractions(error)
    use_model(monkeypatch, fake)

    response = await say(client, "demo", "hi")

    assert response.status_code == 502
    assert "overloaded" not in response.text and "quota" not in response.text
    assert fake.calls
