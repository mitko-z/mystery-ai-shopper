import importlib
import json
import sys
from types import SimpleNamespace

import pytest

import config
from shop import seed
from tool_recorder import recording

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture
def agent(monkeypatch):
    """Import a fresh support_agent with no API key set."""
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    sys.modules.pop("support_agent", None)
    return importlib.import_module("support_agent")


class FakeStep:
    """Stands in for an SDK step: only `model_dump` is used by the agent."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self, **kwargs):
        return dict(self._fields)


class FakeInteractions:
    """Stands in for `client.aio.interactions`; records the calls it receives."""

    def __init__(self, steps):
        self.calls = []
        self._interaction = SimpleNamespace(steps=steps)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._interaction


class ScriptedInteractions:
    """Stands in for `client.aio.interactions`; answers each call with the next scripted turn.

    Each turn is a list of step dicts. Calling the model more often than the
    test scripted fails the test, so a runaway loop cannot hang it.
    """

    def __init__(self, *turns):
        self.calls = []
        self._turns = list(turns)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > len(self._turns):
            raise AssertionError("the model was called more often than the test scripted")
        steps = [FakeStep(**step) for step in self._turns[len(self.calls) - 1]]
        return SimpleNamespace(steps=steps)


def use_fake(agent, monkeypatch, fake):
    """Make the agent talk to `fake` instead of the real Gemini client."""
    client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
    monkeypatch.setattr(agent, "_get_client", lambda: client)


def user_step(text):
    """Build the user step the way run_turn records it."""
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def reply_step(text):
    """Build a model_output step as the model returns it."""
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def call_step(call_id, name, arguments):
    """Build a function_call step as the model returns it."""
    return {"type": "function_call", "id": call_id, "name": name, "arguments": arguments}


def result_of(history, call_id):
    """Return the function_result step answering `call_id`; fail unless there is exactly one."""
    results = [s for s in history if s["type"] == "function_result" and s["call_id"] == call_id]
    assert len(results) == 1
    return results[0]


async def test_import_without_api_key(agent):
    """Importing the module does not need the API key."""
    assert callable(agent.run_turn)


async def test_missing_key_fails_on_first_use(agent):
    """The missing-key error surfaces when the turn runs, not on import."""
    history = []
    with pytest.raises(config.MissingAPIKeyError):
        await agent.run_turn(history, "hi")
    assert history == []


async def test_turn_is_recorded_in_history_only(agent, monkeypatch):
    """The user message and the reply are appended, and nothing is returned."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Hello!"))])
    use_fake(agent, monkeypatch, fake)
    history = []

    result = await agent.run_turn(history, "hi")

    assert result is None
    assert history == [user_step("hi"), reply_step("Hello!")]


async def test_history_is_sent_before_the_new_message(agent, monkeypatch):
    """Earlier turns are passed to the model ahead of the new user message."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Sure."))])
    use_fake(agent, monkeypatch, fake)
    history = [user_step("hi"), reply_step("Hello!")]

    await agent.run_turn(history, "stock of P1?")

    sent = fake.calls[0]["input"]
    assert sent == [user_step("hi"), reply_step("Hello!"), user_step("stock of P1?")]
    assert history == [*sent, reply_step("Sure.")]


async def test_tool_calls_are_recorded_in_history(agent, monkeypatch, capsys):
    """Every step the model produced is kept, tool calls included, in order."""
    thought = {"type": "thought", "signature": "sig"}
    call = {"type": "function_call", "id": "call_1", "name": "check_stock",
            "arguments": {"product_id": "P1"}, "signature": "sig"}
    fake = FakeInteractions(steps=[FakeStep(**thought), FakeStep(**call)])
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.run_turn(history, "stock of P1?")

    assert history == [user_step("stock of P1?"), thought, call]
    assert "check_stock" in capsys.readouterr().out


async def test_history_untouched_when_response_has_no_steps(agent, monkeypatch):
    """A response without steps raises and leaves the history as it was."""
    use_fake(agent, monkeypatch, FakeInteractions(steps=None))
    history = [user_step("hi")]

    with pytest.raises(ValueError):
        await agent.run_turn(history, "again")

    assert history == [user_step("hi")]


# --- get_complete_reply ---

async def test_tool_result_is_sent_back_to_the_model(agent, shop, monkeypatch):
    """A requested tool runs, and the model sees its result on the next turn."""
    fake = ScriptedInteractions(
        [call_step("c1", "check_stock", {"product_id": "P1"})],
        [reply_step("P1 is in stock.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(agent.tools_for(shop), history, "stock of P1?")

    result = result_of(history, "c1")
    assert result["is_error"] is False
    assert json.loads(result["result"])["in_stock"] is True
    assert fake.calls[1]["input"][-1] == result
    assert history[-1] == reply_step("P1 is in stock.")
    # Only the customer's message is a user turn; follow-up turns add no empty one.
    sent_users = [s for s in fake.calls[1]["input"] if s["type"] == "user_input"]
    assert sent_users == [user_step("stock of P1?")]


async def test_every_call_of_a_turn_is_answered(agent, shop, monkeypatch):
    """Parallel calls in one turn each get a result, after all the calls."""
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"}),
         call_step("c2", "check_stock", {"product_id": "P1"})],
        [reply_step("done")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(agent.tools_for(shop), history, "order O1001 and stock of P1?")

    assert [s["type"] for s in history] == [
        "user_input", "function_call", "function_call",
        "function_result", "function_result", "model_output",
    ]
    assert result_of(history, "c1")["is_error"] is False
    assert result_of(history, "c2")["is_error"] is False


# --- unknown tools and bad arguments ---

@pytest.mark.parametrize("name", ["no_such_tool", "_view"])
async def test_unknown_tool_gets_an_error_result(agent, shop, monkeypatch, name):
    """A tool outside the tool list is not run; the model is told, and the loop goes on.

    `_view` exists on the shop but is not a tool, so it must not be callable.
    """
    await shop.refund_order("O1001", 100)
    fake = ScriptedInteractions(
        [call_step("c1", name, {})],
        [reply_step("I cannot do that.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(agent.tools_for(shop), history, "do something odd")

    result = result_of(history, "c1")
    assert result["is_error"] is True
    assert name in json.loads(result["result"])["error"]
    assert fake.calls[1]["input"][-1] == result
    assert history[-1] == reply_step("I cannot do that.")
    assert (await shop.get_order("O1001"))["order"]["refunded"] == 100


@pytest.mark.parametrize(
    "arguments, mentioned",
    [
        ({}, "order_id"),
        ({"order_id": "O1001", "bogus": 1}, "bogus"),
    ],
    ids=["missing-argument", "unexpected-argument"],
)
async def test_bad_arguments_get_an_error_result(agent, shop, monkeypatch, arguments, mentioned):
    """A tool called with wrong arguments reports the error to the model instead of crashing."""
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", arguments)],
        [reply_step("Sorry, let me try again.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(agent.tools_for(shop), history, "where is my order?")

    result = result_of(history, "c1")
    assert result["is_error"] is True
    payload = json.loads(result["result"])
    assert payload["ok"] is False
    assert mentioned in payload["error"]
    assert history[-1] == reply_step("Sorry, let me try again.")


# --- turn limit ---

async def test_tools_are_not_run_on_the_last_turn(agent, shop, monkeypatch):
    """Calls requested on the last turn are refused, not run, and the limit error is raised."""
    monkeypatch.setattr(agent, "MAX_TURNS", 3)
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [call_step("c3", "refund_order", {"order_id": "O1001"})],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    with pytest.raises(agent.TurnLimitError):
        await agent.get_complete_reply(agent.tools_for(shop), history, "refund O1001")

    assert len(fake.calls) == 3
    assert (await shop.get_order("O1001"))["order"]["refunds"] == []
    refused = result_of(history, "c3")
    assert refused["is_error"] is True
    assert "Not executed" in json.loads(refused["result"])["error"]
    # Calls from the earlier turns did run.
    assert result_of(history, "c1")["is_error"] is False
    assert result_of(history, "c2")["is_error"] is False


async def test_history_stays_valid_after_the_limit(agent, shop, monkeypatch):
    """After the limit error every call has exactly one result, so the history can be reused."""
    monkeypatch.setattr(agent, "MAX_TURNS", 2)
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    with pytest.raises(agent.TurnLimitError):
        await agent.get_complete_reply(agent.tools_for(shop), history, "where is my order?")

    call_ids = [s["id"] for s in history if s["type"] == "function_call"]
    result_ids = [s["call_id"] for s in history if s["type"] == "function_result"]
    assert call_ids == result_ids == ["c1", "c2"]
    assert history[-1]["type"] == "function_result"


async def test_tools_run_on_the_turn_before_the_last(agent, shop, monkeypatch):
    """A call on the turn before the last still runs, because the model gets one more turn to report it."""
    monkeypatch.setattr(agent, "MAX_TURNS", 3)
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [call_step("c2", "refund_order", {"order_id": "O1001"})],
        [reply_step("Your order is refunded.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(agent.tools_for(shop), history, "refund O1001")

    assert (await shop.get_order("O1001"))["order"]["refunds"] == [{"amount": 5000}]
    assert result_of(history, "c2")["is_error"] is False
    assert history[-1] == reply_step("Your order is refunded.")


# --- earlier calls ---

async def test_calls_from_earlier_turns_are_not_run_again(agent, shop, monkeypatch):
    """A second customer message on the same history does not repeat the first one's tool calls."""
    fake = ScriptedInteractions(
        [call_step("c1", "refund_order", {"order_id": "O1001", "amount": 1000})],
        [reply_step("Refunded 10.00.")],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [reply_step("It was delivered.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(agent.tools_for(shop), history, "refund 10.00 of O1001")
    await agent.get_complete_reply(agent.tools_for(shop), history, "what is the status of O1001?")

    assert (await shop.get_order("O1001"))["order"]["refunds"] == [{"amount": 1000}]
    result_ids = [s["call_id"] for s in history if s["type"] == "function_result"]
    assert result_ids == ["c1", "c2"]
    assert history[-1] == reply_step("It was delivered.")


# --- call log ---

async def test_failed_call_is_logged_with_its_error(agent, shop, monkeypatch):
    """A call with wrong arguments is logged with the error, and the agent still gets its result."""
    log = []
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {})],
        [reply_step("Sorry, let me try again.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(recording(agent.tools_for(shop), log), history, "where is my order?")

    assert [call.name for call in log] == ["get_order"]
    assert log[0].arguments == {}
    assert log[0].result is None
    assert "order_id" in log[0].error
    assert result_of(history, "c1")["is_error"] is True


async def test_unknown_tool_is_not_in_the_call_log(agent, shop, monkeypatch):
    """The log lists calls that ran a tool. A request for a tool that does not exist is not one."""
    log = []
    fake = ScriptedInteractions(
        [call_step("c1", "no_such_tool", {})],
        [reply_step("I cannot do that.")],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    await agent.get_complete_reply(recording(agent.tools_for(shop), log), history, "do something odd")

    assert log == []
    assert result_of(history, "c1")["is_error"] is True


async def test_call_refused_on_the_last_turn_is_not_in_the_call_log(agent, shop, monkeypatch):
    """A call that is refused because of the turn limit never ran, so it is not logged."""
    monkeypatch.setattr(agent, "MAX_TURNS", 3)
    log = []
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [call_step("c3", "refund_order", {"order_id": "O1001"})],
    )
    use_fake(agent, monkeypatch, fake)
    history = []

    with pytest.raises(agent.TurnLimitError):
        await agent.get_complete_reply(recording(agent.tools_for(shop), log), history, "refund O1001")

    assert [call.name for call in log] == ["get_order", "get_order"]
    assert result_of(history, "c3")["is_error"] is True


async def test_call_log_does_not_change_what_the_agent_sees(agent, monkeypatch):
    """With or without the recorder, the history ends up the same."""
    async def converse(tools):
        fake = ScriptedInteractions(
            [call_step("c1", "refund_order", {"order_id": "O1001", "amount": 1000})],
            [reply_step("Refunded 10.00.")],
        )
        use_fake(agent, monkeypatch, fake)
        history = []
        await agent.get_complete_reply(tools, history, "refund 10.00 of O1001")
        return history

    plain = await converse(agent.tools_for(seed.get_seed_shop()))
    recorded = await converse(recording(agent.tools_for(seed.get_seed_shop()), []))

    assert recorded == plain
