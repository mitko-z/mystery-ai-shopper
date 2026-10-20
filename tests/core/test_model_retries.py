"""A model call that fails with a rate limit or a server error is tried again."""

import asyncio
import random

import httpx
import pytest

from core import agent_loop, config
from core.agent_loop import AgentSpec, ModelError
from core.usage import Usage, track_usage
from tests.fakes import (
    FakeHttpError,
    ScriptedInteractions,
    call_step,
    reply_step,
    stub_tools,
    use_fake,
    user_step,
)

pytestmark = pytest.mark.anyio

SPEC = AgentSpec(system_instruction="you are a test agent", tool_schemas=[])


@pytest.fixture
def pauses(monkeypatch):
    """The pauses between tries, as they are asked for, without waiting. No random extra time."""
    recorded = []

    async def record_pause(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(agent_loop, "_pause", record_pause)
    monkeypatch.setattr(random, "uniform", lambda low, high: low)
    return recorded


# --- which failures are tried again ---

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_a_rate_limit_or_server_error_is_tried_again_after_a_pause(status, monkeypatch, pauses):
    fake = ScriptedInteractions(FakeHttpError(status), [reply_step("Hello!")])
    use_fake(monkeypatch, fake)
    history = []

    with track_usage() as run:
        await agent_loop.run_turn(history, "hi", SPEC)

    assert history == [user_step("hi"), reply_step("Hello!")]
    assert len(fake.calls) == 2
    assert pauses == [config.RETRY_PAUSE_SECONDS]
    assert run.calls == 1  # the failed try got no answer, so it is not a call


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_other_http_errors_are_not_tried_again(status, monkeypatch, pauses):
    fake = ScriptedInteractions(FakeHttpError(status), [reply_step("never reached")])
    use_fake(monkeypatch, fake)

    with pytest.raises(ModelError) as raised:
        await agent_loop.run_turn([], "hi", SPEC)

    assert raised.value.status_code == status
    assert len(fake.calls) == 1
    assert pauses == []


async def test_an_error_that_is_not_an_http_answer_is_not_touched(monkeypatch, pauses):
    fake = ScriptedInteractions(RuntimeError("connection lost"), [reply_step("never reached")])
    use_fake(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="connection lost"):
        await agent_loop.run_turn([], "hi", SPEC)

    assert len(fake.calls) == 1
    assert pauses == []


# --- how often, and how long to wait ---

async def test_the_pause_doubles_with_each_retry(monkeypatch, pauses):
    fake = ScriptedInteractions(
        FakeHttpError(503), FakeHttpError(503), FakeHttpError(503), [reply_step("Hello!")]
    )
    use_fake(monkeypatch, fake)

    await agent_loop.run_turn([], "hi", SPEC)

    assert pauses == [2.0, 4.0, 8.0]
    assert len(fake.calls) == config.MODEL_CALL_RETRIES + 1


async def test_after_the_last_retry_the_call_fails_with_a_model_error(monkeypatch, pauses):
    errors = [FakeHttpError(503) for _ in range(config.MODEL_CALL_RETRIES + 1)]
    fake = ScriptedInteractions(*errors, [reply_step("never reached")])
    use_fake(monkeypatch, fake)
    history = [user_step("earlier")]

    with track_usage() as run:
        with pytest.raises(ModelError) as raised:
            await agent_loop.run_turn(history, "hi", SPEC)

    assert raised.value.status_code == 503
    assert raised.value.__cause__ is errors[-1]
    assert len(fake.calls) == config.MODEL_CALL_RETRIES + 1
    assert len(pauses) == config.MODEL_CALL_RETRIES
    assert history == [user_step("earlier")]  # a failed turn leaves the history alone
    assert run == Usage()  # no try got an answer, so nothing is counted


@pytest.mark.parametrize(
    "retry_after, expected_pause",
    [
        (1, 2.0),  # shorter than the usual pause: the usual pause
        (5, 5.0),  # longer: wait as long as the server asks
        (config.MAX_RETRY_AFTER_SECONDS, config.MAX_RETRY_AFTER_SECONDS),  # the longest we accept
    ],
)
async def test_retry_after_sets_the_minimum_pause(retry_after, expected_pause, monkeypatch, pauses):
    fake = ScriptedInteractions(FakeHttpError(429, retry_after=retry_after), [reply_step("Hello!")])
    use_fake(monkeypatch, fake)

    await agent_loop.run_turn([], "hi", SPEC)

    assert pauses == [expected_pause]


async def test_a_retry_after_that_is_too_long_ends_the_call_at_once(monkeypatch, pauses):
    """A daily quota asks for hours. Waiting would make the run look hung."""
    fake = ScriptedInteractions(FakeHttpError(429, retry_after=3600), [reply_step("never reached")])
    use_fake(monkeypatch, fake)

    with pytest.raises(ModelError) as raised:
        await agent_loop.run_turn([], "hi", SPEC)

    assert raised.value.status_code == 429
    assert len(fake.calls) == 1
    assert pauses == []


async def test_a_retry_after_that_is_not_a_number_is_ignored(monkeypatch, pauses):
    fake = ScriptedInteractions(
        FakeHttpError(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT"), [reply_step("Hello!")]
    )
    use_fake(monkeypatch, fake)

    await agent_loop.run_turn([], "hi", SPEC)

    assert pauses == [config.RETRY_PAUSE_SECONDS]


async def test_some_random_time_is_added_to_the_pause(monkeypatch, pauses):
    """So that runs which failed together do not all retry at the same moment."""
    monkeypatch.setattr(random, "uniform", lambda low, high: high)
    use_fake(monkeypatch, ScriptedInteractions(FakeHttpError(503), [reply_step("Hello!")]))

    await agent_loop.run_turn([], "hi", SPEC)

    assert pauses == [config.RETRY_PAUSE_SECONDS * (1 + config.RETRY_JITTER)]


# --- how retries meet the rest ---

async def test_a_retried_call_uses_up_one_call_of_the_limit(monkeypatch, pauses):
    use_fake(monkeypatch, ScriptedInteractions(FakeHttpError(503), [reply_step("Hello!")]))

    with track_usage(max_calls=1) as run:
        await agent_loop.run_turn([], "hi", SPEC)

    assert run.calls == 1


async def test_a_tool_loop_goes_on_after_a_retried_call(monkeypatch, pauses):
    effects = []
    fake = ScriptedInteractions(
        [call_step("c1", "lookup", {"key": "A"})],
        FakeHttpError(503),
        [reply_step("A is found.")],
    )
    use_fake(monkeypatch, fake)
    history = []

    await agent_loop.get_complete_reply(stub_tools(effects), history, "look up A", SPEC)

    assert history[-1] == reply_step("A is found.")
    assert pauses == [config.RETRY_PAUSE_SECONDS]


async def test_waiting_between_tries_can_be_cancelled_and_is_not_a_cancelled_call(monkeypatch):
    """Nothing is in flight during a pause, so a timeout there costs no call."""
    async def wait_forever(seconds):
        await asyncio.Event().wait()

    monkeypatch.setattr(agent_loop, "_pause", wait_forever)
    use_fake(monkeypatch, ScriptedInteractions(FakeHttpError(503), [reply_step("never reached")]))

    with track_usage() as run:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await agent_loop.run_turn([], "hi", SPEC)

    assert run == Usage()


# --- the real SDK ---
# The SDK has its own retrying, and its own error classes. These tests run the real
# client against a fake server, to check that the SDK does not retry on top of us
# and that its errors are the kind this module understands.

def sdk_client(monkeypatch, status, requests, headers=None):
    """Make `run_turn` use the real SDK client, talking to a server that always answers `status`."""
    def server(request):
        requests.append(request)
        return httpx.Response(status, headers=headers or {}, json={"error": {"code": status, "message": "boom"}})

    client = agent_loop._build_client(
        "test-key", httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
    )
    monkeypatch.setattr(agent_loop, "_get_client", lambda: client)


@pytest.mark.parametrize("status", [429, 503])
async def test_the_sdk_sends_one_request_per_try(status, monkeypatch, pauses):
    requests = []
    sdk_client(monkeypatch, status, requests)

    with pytest.raises(ModelError) as raised:
        await agent_loop.run_turn([], "hi", SPEC)

    assert raised.value.status_code == status
    assert len(requests) == config.MODEL_CALL_RETRIES + 1
    assert len(pauses) == config.MODEL_CALL_RETRIES


async def test_the_sdk_does_not_sleep_for_a_long_retry_after(monkeypatch, pauses):
    requests = []
    sdk_client(monkeypatch, 429, requests, headers={"Retry-After": "3600"})

    with pytest.raises(ModelError) as raised:
        async with asyncio.timeout(5):  # a client that slept for the header would hit this
            await agent_loop.run_turn([], "hi", SPEC)

    assert raised.value.status_code == 429
    assert len(requests) == 1
    assert pauses == []


async def test_the_sdk_does_not_retry_a_client_error(monkeypatch, pauses):
    requests = []
    sdk_client(monkeypatch, 400, requests)

    with pytest.raises(ModelError) as raised:
        await agent_loop.run_turn([], "hi", SPEC)

    assert raised.value.status_code == 400
    assert len(requests) == 1
