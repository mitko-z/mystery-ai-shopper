import datetime

import pytest

from tool_recorder import recording

pytestmark = pytest.mark.anyio       # every test here is async


@pytest.fixture
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"

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