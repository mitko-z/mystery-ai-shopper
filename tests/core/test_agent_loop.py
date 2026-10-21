import asyncio
import json
import logging
import os
import subprocess
import sys

import pytest

from core import agent_loop, config
from core.agent_loop import AgentSpec
from core.usage import Usage, track_usage
from tests.fakes import (
    FakeInteractions,
    FakeStep,
    HangingInteractions,
    ScriptedInteractions,
    Turn,
    call_step,
    reply_step,
    reported_usage,
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


# --- usage ---

class FailingInteractions:
    """Stands in for `client.aio.interactions`; every call raises, as a 429 would."""

    async def create(self, **kwargs):
        raise RuntimeError("quota exhausted")


async def test_a_model_call_is_counted_with_its_tokens(agent, monkeypatch):
    """The usage report of a response is added to the run's usage."""
    use_fake(monkeypatch, FakeInteractions(
        steps=[FakeStep(**reply_step("Hello!"))],
        usage=reported_usage(input=100, output=20, thought=30, total=150),
    ))

    with track_usage() as run:
        await agent.run_turn([], "hi", SPEC)

    assert run == Usage(
        calls=1, input_tokens=100, output_tokens=20, thought_tokens=30, total_tokens=150
    )


async def test_a_response_without_usage_is_counted_and_flagged(agent, monkeypatch):
    """A response that carries no usage report is still a call, and is flagged."""
    use_fake(monkeypatch, FakeInteractions(steps=[FakeStep(**reply_step("Hello!"))]))

    with track_usage() as run:
        await agent.run_turn([], "hi", SPEC)

    assert run == Usage(calls=1, calls_without_usage=1)


async def test_a_response_without_steps_is_still_counted(agent, monkeypatch):
    """The model answered and was paid for, even if the answer is unusable."""
    use_fake(monkeypatch, FakeInteractions(steps=None, usage=reported_usage(input=10, total=10)))

    with track_usage() as run:
        with pytest.raises(ValueError):
            await agent.run_turn([], "hi", SPEC)

    assert run == Usage(calls=1, input_tokens=10, total_tokens=10)


async def test_a_failed_model_call_is_not_counted(agent, monkeypatch):
    """A call that raises got no response, so it is not counted."""
    use_fake(monkeypatch, FailingInteractions())

    with track_usage() as run:
        with pytest.raises(RuntimeError):
            await agent.run_turn([], "hi", SPEC)

    assert run == Usage()


async def test_every_turn_of_a_tool_loop_is_counted(agent, tools, monkeypatch):
    """One customer message that needs two tool calls makes three model calls."""
    use_fake(monkeypatch, ScriptedInteractions(
        Turn([call_step("c1", "lookup", {"key": "A"})], reported_usage(input=10, output=1, total=11)),
        Turn([call_step("c2", "lookup", {"key": "B"})], reported_usage(input=20, output=1, total=21)),
        Turn([reply_step("Both found.")], reported_usage(input=30, output=5, total=35)),
    ))

    with track_usage() as run:
        await agent.get_complete_reply(tools, [], "look up A and B", SPEC)

    assert run == Usage(calls=3, input_tokens=60, output_tokens=7, total_tokens=67)


async def test_a_call_cancelled_by_a_timeout_is_counted_apart(agent, monkeypatch):
    """The call in flight when a timeout fires got no response, so its tokens are unknown."""
    use_fake(monkeypatch, HangingInteractions())
    history = []

    with track_usage() as run:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await agent.run_turn(history, "hi", SPEC)

    assert run == Usage(calls_cancelled=1)
    assert history == []


# --- the call limit ---

async def test_a_run_that_used_up_its_calls_cannot_make_another(agent, monkeypatch):
    """The limit is checked before the model is called, so a refused call costs nothing."""
    fake = ScriptedInteractions([reply_step("one")], [reply_step("two")])
    use_fake(monkeypatch, fake)
    history = []

    with track_usage(max_calls=2) as run:
        await agent.run_turn(history, "a", SPEC)
        await agent.run_turn(history, "b", SPEC)
        with pytest.raises(agent.CallLimitError, match="limit of 2 model calls"):
            await agent.run_turn(history, "c", SPEC)

    assert len(fake.calls) == 2
    assert run.calls == 2
    assert history == [user_step("a"), reply_step("one"), user_step("b"), reply_step("two")]


async def test_a_cancelled_call_counts_toward_the_limit(agent, monkeypatch):
    """A call that a timeout cancelled was probably billed, so it uses up a call too."""
    use_fake(monkeypatch, HangingInteractions())

    with track_usage(max_calls=1):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await agent.run_turn([], "hi", SPEC)
        with pytest.raises(agent.CallLimitError):
            # The fake never answers: without a limit this call would hang, so guard it
            # and let a missing limit fail the test instead.
            async with asyncio.timeout(1):
                await agent.run_turn([], "again", SPEC)


async def test_the_limit_stops_a_tool_loop_and_says_why(agent, tools, monkeypatch):
    """The caller sees the call limit, not a turn-limit error that would hide the reason."""
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        [call_step("c2", "lookup", {"key": "B"})],
        [reply_step("Both found.")],
    )
    use_fake(monkeypatch, fake)

    with track_usage(max_calls=2):
        with pytest.raises(agent.CallLimitError):
            await agent.get_complete_reply(tools, [], "look up A and B", SPEC)

    assert len(fake.calls) == 2


async def test_without_a_tracked_run_there_is_no_limit(agent, monkeypatch):
    """Code that does not track usage, such as the shop server on its own, is never refused."""
    fake = FakeInteractions(steps=[FakeStep(**reply_step("Hello!"))])
    use_fake(monkeypatch, fake)
    calls = config.DEFAULT_CALLS_PER_RUN + 1

    for _ in range(calls):
        await agent.run_turn([], "hi", SPEC)

    assert len(fake.calls) == calls


# --- checking the key before the first call ---

async def test_check_api_key_fails_at_once_without_a_key(agent):
    """The batch uses it to stop before any run starts."""
    with pytest.raises(config.MissingAPIKeyError):
        agent.check_api_key()


async def test_check_api_key_passes_with_a_key(agent, monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "test-key")
    try:
        agent.check_api_key()
    finally:
        agent._get_client.cache_clear()  # do not leave a real client for the other tests


async def test_check_api_key_asks_for_the_client_the_model_calls_use(agent, monkeypatch):
    """So a test that replaces the client with a fake needs no key."""
    asked = []
    monkeypatch.setattr(agent, "_get_client", lambda: asked.append("client"))

    agent.check_api_key()

    assert asked == ["client"]
