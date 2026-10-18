import json
import logging
import os
import subprocess
import sys

import pytest

from core import agent_loop, config
from core.agent_loop import AgentSpec
from tests.fakes import (
    FakeInteractions,
    FakeStep,
    ScriptedInteractions,
    call_step,
    reply_step,
    result_of,
    stub_tools,
    use_fake,
    user_step,
)

pytestmark = pytest.mark.anyio

SPEC = AgentSpec(
    system_instruction="you are a test agent",
    tool_schemas=[{"type": "function", "name": "lookup"}],
)


@pytest.fixture
def agent(monkeypatch):
    """The agent_loop module, with no API key set."""
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    return agent_loop


@pytest.fixture
def effects():
    """What the `record` tool did: one entry per run."""
    return []


@pytest.fixture
def tools(effects):
    """A tool table that needs no shop."""
    return stub_tools(effects)


async def test_import_without_api_key():
    """Importing the module does not need the API key."""
    env = {k: v for k, v in os.environ.items() if k != config.API_KEY_ENV_VAR}
    subprocess.run([sys.executable, "-c", "import core.agent_loop"], env=env, check=True)


async def test_missing_key_fails_on_first_use(agent):
    """The missing-key error surfaces when the turn runs, not on import."""
    history = []
    with pytest.raises(config.MissingAPIKeyError):
        await agent.run_turn(history, "hi", SPEC)
    assert history == []


async def test_turn_is_recorded_in_history_only(agent, monkeypatch):
    """The user message and the reply are appended, and nothing is returned."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Hello!"))])
    use_fake(monkeypatch, fake)
    history = []

    result = await agent.run_turn(history, "hi", SPEC)

    assert result is None
    assert history == [user_step("hi"), reply_step("Hello!")]


async def test_spec_is_sent_to_the_model(agent, monkeypatch):
    """The model, system instruction and tool schemas come from the agent's spec."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Hello!"))])
    use_fake(monkeypatch, fake)
    spec = AgentSpec(
        system_instruction="you are another agent",
        tool_schemas=[{"type": "function", "name": "other"}],
        model="some-model",
    )

    await agent.run_turn([], "hi", spec)

    sent = fake.calls[0]
    assert sent["model"] == "some-model"
    assert sent["system_instruction"] == "you are another agent"
    assert sent["tools"] == [{"type": "function", "name": "other"}]


async def test_history_is_sent_before_the_new_message(agent, monkeypatch):
    """Earlier turns are passed to the model ahead of the new user message."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Sure."))])
    use_fake(monkeypatch, fake)
    history = [user_step("hi"), reply_step("Hello!")]

    await agent.run_turn(history, "stock of P1?", SPEC)

    sent = fake.calls[0]["input"]
    assert sent == [user_step("hi"), reply_step("Hello!"), user_step("stock of P1?")]
    assert history == [*sent, reply_step("Sure.")]


async def test_tool_calls_are_recorded_in_history(agent, monkeypatch, caplog):
    """Every step the model produced is kept, tool calls included, in order."""
    thought = {"type": "thought", "signature": "sig"}
    call = {"type": "function_call", "id": "call_1", "name": "lookup",
            "arguments": {"key": "A"}, "signature": "sig"}
    fake = FakeInteractions(steps=[FakeStep(**thought), FakeStep(**call)])
    use_fake(monkeypatch, fake)
    history = []

    with caplog.at_level(logging.DEBUG, logger="core.agent_loop"):
        await agent.run_turn(history, "look up A", SPEC)

    assert history == [user_step("look up A"), thought, call]
    assert "lookup" in caplog.text


async def test_history_untouched_when_response_has_no_steps(agent, monkeypatch):
    """A response without steps raises and leaves the history as it was."""
    use_fake(monkeypatch, FakeInteractions(steps=None))
    history = [user_step("hi")]

    with pytest.raises(ValueError):
        await agent.run_turn(history, "again", SPEC)

    assert history == [user_step("hi")]


# --- get_complete_reply ---

async def test_tool_result_is_sent_back_to_the_model(agent, tools, monkeypatch):
    """A requested tool runs, and the model sees its result on the next turn."""
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        [reply_step("A is found.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent.get_complete_reply(tools, history, "look up A", SPEC)

    result = result_of(history, "c1")
    assert result["is_error"] is False
    assert json.loads(result["result"])["value"] == "value of A"
    assert fake.calls[1]["input"][-1] == result
    assert history[-1] == reply_step("A is found.")
    # Only the customer's message is a user turn; follow-up turns add no empty one.
    sent_users = [s for s in fake.calls[1]["input"] if s["type"] == "user_input"]
    assert sent_users == [user_step("look up A")]


async def test_every_call_of_a_turn_is_answered(agent, tools, monkeypatch):
    """Parallel calls in one turn each get a result, after all the calls."""
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"}),
         call_step("c2", "lookup", {"key": "B"})],
        [reply_step("done")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent.get_complete_reply(tools, history, "look up A and B", SPEC)

    assert [s["type"] for s in history] == [
        "user_input", "function_call", "function_call",
        "function_result", "function_result", "model_output",
    ]
    assert result_of(history, "c1")["is_error"] is False
    assert result_of(history, "c2")["is_error"] is False


# --- unknown tools and bad arguments ---

async def test_unknown_tool_gets_an_error_result(agent, tools, effects, monkeypatch):
    """A tool outside the tool table is not run; the model is told, and the loop goes on."""
    fake = ScriptedInteractions(
        [call_step("c1", "no_such_tool", {})],
        [reply_step("I cannot do that.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent.get_complete_reply(tools, history, "do something odd", SPEC)

    result = result_of(history, "c1")
    assert result["is_error"] is True
    assert "no_such_tool" in json.loads(result["result"])["error"]
    assert fake.calls[1]["input"][-1] == result
    assert history[-1] == reply_step("I cannot do that.")
    assert effects == []


@pytest.mark.parametrize(
    "arguments, mentioned",
    [
        ({}, "key"),
        ({"key": "A", "bogus": 1}, "bogus"),
    ],
    ids=["missing-argument", "unexpected-argument"],
)
async def test_bad_arguments_get_an_error_result(agent, tools, monkeypatch, arguments, mentioned):
    """A tool called with wrong arguments reports the error to the model instead of crashing."""
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", arguments)],
        [reply_step("Sorry, let me try again.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent.get_complete_reply(tools, history, "look something up", SPEC)

    result = result_of(history, "c1")
    assert result["is_error"] is True
    payload = json.loads(result["result"])
    assert payload["ok"] is False
    assert mentioned in payload["error"]
    assert history[-1] == reply_step("Sorry, let me try again.")


# --- turn limit ---

async def test_tools_are_not_run_on_the_last_turn(agent, tools, effects, monkeypatch):
    """Calls requested on the last turn are refused, not run, and the limit error is raised."""
    monkeypatch.setattr(agent, "MAX_TURNS", 3)
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        [call_step("c2", "lookup", {"key": "A"})],
        [call_step("c3", "record", {"key": "A"})],
    )
    use_fake(monkeypatch, fake)
    history = []

    with pytest.raises(agent.TurnLimitError):
        await agent.get_complete_reply(tools, history, "record A", SPEC)

    assert len(fake.calls) == 3
    assert effects == []
    refused = result_of(history, "c3")
    assert refused["is_error"] is True
    assert "Not executed" in json.loads(refused["result"])["error"]
    # Calls from the earlier turns did run.
    assert result_of(history, "c1")["is_error"] is False
    assert result_of(history, "c2")["is_error"] is False


async def test_history_stays_valid_after_the_limit(agent, tools, monkeypatch):
    """After the limit error every call has exactly one result, so the history can be reused."""
    monkeypatch.setattr(agent, "MAX_TURNS", 2)
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        [call_step("c2", "lookup", {"key": "A"})],
    )
    use_fake(monkeypatch, fake)
    history = []

    with pytest.raises(agent.TurnLimitError):
        await agent.get_complete_reply(tools, history, "look up A", SPEC)

    call_ids = [s["id"] for s in history if s["type"] == "function_call"]
    result_ids = [s["call_id"] for s in history if s["type"] == "function_result"]
    assert call_ids == result_ids == ["c1", "c2"]
    assert history[-1]["type"] == "function_result"


async def test_tools_run_on_the_turn_before_the_last(agent, tools, effects, monkeypatch):
    """A call on the turn before the last still runs, because the model gets one more turn to report it."""
    monkeypatch.setattr(agent, "MAX_TURNS", 3)
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        [call_step("c2", "record", {"key": "A"})],
        [reply_step("Recorded A.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent.get_complete_reply(tools, history, "record A", SPEC)

    assert effects == [("A", None)]
    assert result_of(history, "c2")["is_error"] is False
    assert history[-1] == reply_step("Recorded A.")


# --- earlier calls ---

async def test_calls_from_earlier_turns_are_not_run_again(agent, tools, effects, monkeypatch):
    """A second customer message on the same history does not repeat the first one's tool calls."""
    fake = ScriptedInteractions(
        [call_step("c1", "record", {"key": "A", "amount": 1000})],
        [reply_step("Recorded 1000.")],
        [call_step("c2", "lookup", {"key": "A"})],
        [reply_step("A is found.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent.get_complete_reply(tools, history, "record 1000 for A", SPEC)
    await agent.get_complete_reply(tools, history, "look up A", SPEC)

    assert effects == [("A", 1000)]
    result_ids = [s["call_id"] for s in history if s["type"] == "function_result"]
    assert result_ids == ["c1", "c2"]
    assert history[-1] == reply_step("A is found.")


# --- final text ---

def test_final_text_is_the_model_output_at_the_end():
    """Thoughts, calls and results of the turn are skipped; the closing text parts are joined as they are."""
    history = [
        user_step("look up A"),
        {"type": "thought", "signature": "sig"},
        call_step("c1", "lookup", {"key": "A"}),
        {"type": "function_result", "call_id": "c1", "name": "lookup", "result": "{}", "is_error": False},
        reply_step("Found. "),
        reply_step("Value of A."),
    ]

    assert agent_loop.final_text(history) == "Found. Value of A."


def test_final_text_ignores_model_output_of_earlier_turns():
    """Only the model_output steps at the very end belong to the latest answer."""
    history = [user_step("hi"), reply_step("Hello!"), user_step("bye"), reply_step("Bye!")]

    assert agent_loop.final_text(history) == "Bye!"


def test_final_text_is_empty_without_text_parts():
    """A final step with no text parts gives an empty string, not an error."""
    assert agent_loop.final_text([user_step("hi"), {"type": "model_output"}]) == ""
