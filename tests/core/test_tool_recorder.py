import asyncio
import datetime
import decimal
import json

import pytest

from core import agent_loop
from core.agent_loop import AgentSpec
from core.tool_recorder import ToolCall, recording
from tests.fakes import ScriptedInteractions, call_step, reply_step, result_of, stub_tools, use_fake

pytestmark = pytest.mark.anyio       # every test here is async

SPEC = AgentSpec(system_instruction="you are a test agent", tool_schemas=[])


async def test_call_is_passed_through_and_logged():
    """The wrapped tool returns what the real one returns, and the
    call is logged."""
    async def add(x, y):
        return x + y

    log = []
    tools = recording({"add": add}, log)

    result = await tools["add"](x=2, y=3)

    assert result == 5
    assert len(log) == 1
    assert log[0].name == "add" and log[0].arguments == {"x": 2, "y": 3} and log[0].result == 5
    assert log[0].started_at is not None and log[0].started_at.tzinfo == datetime.timezone.utc
    assert log[0].duration >= 0


async def test_calls_are_logged_in_the_order_they_happen():
    """The log lists the calls oldest first."""
    async def add(x, y):
        return x + y

    log = []
    tools = recording({"add": add}, log)

    await tools["add"](x=1, y=1)
    await tools["add"](x=2, y=2)
    await tools["add"](x=3, y=3)

    assert [call.result for call in log] == [2, 4, 6]
    assert [call.started_at for call in log] == sorted(call.started_at for call in log)


async def test_failing_tool_is_logged_and_the_error_is_raised():
    """A tool that raises is still logged, with its error, and the caller sees the exception."""
    async def boom(x):
        raise ValueError(f"bad x: {x}")

    log = []
    tools = recording({"boom": boom}, log)

    with pytest.raises(ValueError, match="bad x: 1"):
        await tools["boom"](x=1)

    assert len(log) == 1
    assert log[0].name == "boom"
    assert log[0].arguments == {"x": 1}
    assert log[0].result is None
    assert log[0].error == "bad x: 1"


async def test_cancelled_call_is_logged_and_the_cancellation_is_raised():
    """wait_for cancels a slow call by raising CancelledError inside it. That is not an
    Exception, but the call is still logged, and the cancellation still reaches the caller."""
    async def slow():
        await asyncio.sleep(10)

    log = []
    tools = recording({"slow": slow}, log)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tools["slow"](), timeout=0.01)

    assert len(log) == 1
    assert log[0].name == "slow"
    assert log[0].result is None
    assert log[0].error == "CancelledError"
    assert 0 < log[0].duration < 5


async def test_call_with_wrong_arguments_is_logged_and_the_error_is_raised():
    """Wrong arguments make the tool raise a TypeError. That call is logged too."""
    async def add(x, y):
        return x + y

    log = []
    tools = recording({"add": add}, log)

    with pytest.raises(TypeError):
        await tools["add"](x=1)

    assert len(log) == 1
    assert log[0].arguments == {"x": 1}
    assert "y" in log[0].error


async def test_each_wrapped_table_logs_to_its_own_log():
    """Two tables built from the same tools do not write to each other's log."""
    async def ping():
        return "pong"

    log_a, log_b = [], []
    tools_a = recording({"ping": ping}, log_a)
    tools_b = recording({"ping": ping}, log_b)

    await tools_a["ping"]()
    await tools_b["ping"]()
    await tools_b["ping"]()

    assert len(log_a) == 1
    assert len(log_b) == 2


async def test_wrapping_keeps_the_tool_names_and_the_original_table():
    """The wrapped table has the same tools, and the table that was passed in is not changed."""
    async def ping():
        return "pong"

    original = {"ping": ping}
    tools = recording(original, [])

    assert set(tools) == {"ping"}
    assert original == {"ping": ping}
    assert tools["ping"] is not ping


# --- with the agent loop ---

async def test_failed_call_is_logged_with_its_error(monkeypatch):
    """A call with wrong arguments is logged with the error, and the agent still gets its result."""
    log = []
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {})],
        [reply_step("Sorry, let me try again.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent_loop.get_complete_reply(recording(stub_tools([]), log), history, "look something up", SPEC)

    assert [call.name for call in log] == ["lookup"]
    assert log[0].arguments == {}
    assert log[0].result is None
    assert "key" in log[0].error
    assert result_of(history, "c1")["is_error"] is True


async def test_unknown_tool_is_not_in_the_call_log(monkeypatch):
    """The log lists calls that ran a tool. A request for a tool that does not exist is not one."""
    log = []
    fake = ScriptedInteractions(
        [call_step("c1", "no_such_tool", {})],
        [reply_step("I cannot do that.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent_loop.get_complete_reply(recording(stub_tools([]), log), history, "do something odd", SPEC)

    assert log == []
    assert result_of(history, "c1")["is_error"] is True


async def test_call_refused_on_the_last_turn_is_not_in_the_call_log(monkeypatch):
    """A call that is refused because of the turn limit never ran, so it is not logged."""
    monkeypatch.setattr(agent_loop, "MAX_TURNS", 3)
    log = []
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        [call_step("c2", "lookup", {"key": "A"})],
        [call_step("c3", "record", {"key": "A"})],
    )
    use_fake(monkeypatch, fake)
    history = []

    with pytest.raises(agent_loop.TurnLimitError):
        await agent_loop.get_complete_reply(recording(stub_tools([]), log), history, "record A", SPEC)

    assert [call.name for call in log] == ["lookup", "lookup"]
    assert result_of(history, "c3")["is_error"] is True


async def test_call_log_does_not_change_what_the_agent_sees(monkeypatch):
    """With or without the recorder, the history ends up the same."""
    async def converse(tools):
        fake = ScriptedInteractions(
            [call_step("c1", "record", {"key": "A", "amount": 1000})],
            [reply_step("Recorded 1000.")],
        )
        use_fake(monkeypatch, fake)
        history = []
        await agent_loop.get_complete_reply(tools, history, "record 1000 for A", SPEC)
        return history

    plain = await converse(stub_tools([]))
    recorded = await converse(recording(stub_tools([]), []))

    assert recorded == plain


# --- the JSON-safe form ---

STARTED = datetime.datetime(2026, 10, 20, 9, 30, 0, 123456, tzinfo=datetime.timezone.utc)


def make_call(**overrides):
    """A tool call with placeholder values; `overrides` replace single fields."""
    fields = {
        "name": "get_order",
        "arguments": {"order_id": "O1001"},
        "result": {"ok": True, "status": "shipped"},
        "started_at": STARTED,
        "duration": 0.25,
    }
    return ToolCall(**{**fields, **overrides})


def test_a_call_becomes_a_plain_dict():
    as_dict = make_call().to_dict()

    assert as_dict == {
        "name": "get_order",
        "arguments": {"order_id": "O1001"},
        "result": {"ok": True, "status": "shipped"},
        "started_at": "2026-10-20T09:30:00.123456+00:00",
        "duration": 0.25,
        "error": None,
    }
    assert json.loads(json.dumps(as_dict)) == as_dict


def test_a_failed_call_keeps_its_error():
    as_dict = make_call(result=None, error="ValueError: bad x").to_dict()

    assert as_dict["result"] is None
    assert as_dict["error"] == "ValueError: bad x"


@pytest.mark.parametrize("result", [5, "text", None, True, [1, "a", None], {"a": {"b": [1.5]}}])
def test_a_result_that_json_can_hold_is_kept_as_it_is(result):
    assert make_call(result=result).to_dict()["result"] == result


def test_a_tuple_becomes_a_list():
    assert make_call(result=("a", 1)).to_dict()["result"] == ["a", 1]


def test_what_json_cannot_hold_becomes_text():
    class Order:
        def __str__(self):
            return "order O1001"

    result = {
        "when": STARTED,
        "price": decimal.Decimal("9.99"),
        "tags": {"fragile"},
        "order": Order(),
        "nested": [{"when": STARTED}],
    }

    as_dict = make_call(result=result).to_dict()

    assert as_dict["result"] == {
        "when": "2026-10-20T09:30:00.123456+00:00",
        "price": "9.99",
        "tags": "{'fragile'}",
        "order": "order O1001",
        "nested": [{"when": "2026-10-20T09:30:00.123456+00:00"}],
    }
    json.dumps(as_dict)


def test_arguments_are_made_safe_in_the_same_way():
    as_dict = make_call(arguments={"since": STARTED}).to_dict()

    assert as_dict["arguments"] == {"since": "2026-10-20T09:30:00.123456+00:00"}


def test_a_result_that_cannot_be_converted_piece_by_piece_becomes_text():
    """A dict with tuple keys is not JSON at all. It must not stop the file from being written."""
    as_dict = make_call(result={("a", 1): "x"}).to_dict()

    assert as_dict["result"] == "{('a', 1): 'x'}"
    json.dumps(as_dict)


def test_a_result_that_contains_itself_becomes_text():
    loop = {}
    loop["me"] = loop

    as_dict = make_call(result=loop).to_dict()

    assert isinstance(as_dict["result"], str)
    json.dumps(as_dict)


def test_the_dict_is_a_copy_that_later_changes_do_not_reach():
    call = make_call(arguments={"order_id": "O1001"})
    as_dict = call.to_dict()

    call.arguments["order_id"] = "changed"

    assert as_dict["arguments"] == {"order_id": "O1001"}


async def test_a_recorded_log_can_be_written_as_json():
    async def get_order(order_id):
        return {"ok": True, "order_id": order_id, "placed": STARTED}

    log = []
    tools = recording({"get_order": get_order}, log)
    await tools["get_order"](order_id="O1001")

    written = json.loads(json.dumps([call.to_dict() for call in log]))

    assert written[0]["name"] == "get_order"
    assert written[0]["result"]["placed"] == "2026-10-20T09:30:00.123456+00:00"
