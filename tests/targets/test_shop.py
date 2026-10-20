import json

import pytest

from core import agent_loop, config
from evaluator.target import TargetError, ToolCallSource
from shop import app as shop_app
from targets import shop as shop_target
from tests.fakes import ScriptedInteractions, call_step, reply_step, use_fake, user_step

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def fresh_state():
    """Forget the shop's stored conversations before every test."""
    shop_app._conversations.clear()


async def test_send_returns_the_support_agents_reply(monkeypatch):
    """A message sent through the target reaches the support agent, and its answer comes back."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("Hello! How can I help?")]))

    async with shop_target.open_target() as target:
        reply = await target.send("hi")

    assert reply == "Hello! How can I help?"


async def test_one_run_is_one_conversation(monkeypatch):
    """Later messages of a run see the earlier ones."""
    fake = ScriptedInteractions([reply_step("Hello!")], [reply_step("Sure.")])
    use_fake(monkeypatch, fake)

    async with shop_target.open_target() as target:
        await target.send("one")
        await target.send("two")

    assert fake.calls[1]["input"] == [user_step("one"), reply_step("Hello!"), user_step("two")]


async def test_runs_do_not_share_a_shop(monkeypatch):
    """A refund made in one run is not seen in the next one."""
    fake = ScriptedInteractions(
        [call_step("c1", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded.")],
        [call_step("c2", "get_order", {"order_id": "O1001"})],
        [reply_step("Nothing refunded.")],
    )
    use_fake(monkeypatch, fake)

    async with shop_target.open_target() as first:
        await first.send("refund O1001")
    async with shop_target.open_target() as second:
        await second.send("what is the status of O1001?")

    seen_by_second = json.loads(fake.calls[3]["input"][-1]["result"])
    assert seen_by_second["order"]["refunded"] == 0


async def test_a_shop_failure_is_a_target_error(monkeypatch):
    """The shop's error status is reported as a target error."""
    def no_key():
        raise config.MissingAPIKeyError("not set")

    monkeypatch.setattr(agent_loop, "_get_client", no_key)

    async with shop_target.open_target() as target:
        with pytest.raises(TargetError, match="503"):
            await target.send("hi")


def test_refund_scenario_is_about_order_o1002():
    """The scenario is registered under its own name and mentions the order in both persona and goal."""
    scenario = shop_target.SCENARIOS["refund_o1002"]

    assert scenario.name == "refund_o1002"
    assert "O1002" in scenario.persona
    assert "O1002" in scenario.goal


async def test_the_shop_target_offers_its_tool_call_log():
    """The stand-in keeps a log, so this target can be asked for it."""
    async with shop_target.open_target() as target:
        assert isinstance(target, ToolCallSource)


async def test_the_tool_call_log_is_empty_before_anything_is_sent():
    """No conversation yet is an empty log, not a missing one."""
    async with shop_target.open_target() as target:
        assert target.tool_calls() == []


async def test_the_tool_call_log_lists_what_the_support_agent_did(monkeypatch):
    """After a refund the log shows the calls in order, with their results."""
    fake = ScriptedInteractions(
        [call_step("c1", "get_order", {"order_id": "O1001"})],
        [call_step("c2", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded.")],
    )
    use_fake(monkeypatch, fake)

    async with shop_target.open_target() as target:
        await target.send("refund O1001")
        calls = target.tool_calls()

    assert [(call.name, call.arguments) for call in calls] == [
        ("get_order", {"order_id": "O1001"}),
        ("refund_order", {"order_id": "O1001"}),
    ]
    assert calls[1].result["ok"] is True


async def test_each_run_has_its_own_tool_call_log(monkeypatch):
    """What one run's agent did does not show in another run's log."""
    fake = ScriptedInteractions(
        [call_step("c1", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded.")],
        [reply_step("Hello!")],
    )
    use_fake(monkeypatch, fake)

    async with shop_target.open_target() as first:
        await first.send("refund O1001")
    async with shop_target.open_target() as second:
        await second.send("hi")
        assert second.tool_calls() == []


def test_refund_scenario_has_written_expectations():
    """A judge needs written expectations to compare the transcript with, including things the agent must not do."""
    scenario = shop_target.SCENARIOS["refund_o1002"]

    assert len(scenario.expectations) >= 2
    assert any("must not" in expectation for expectation in scenario.expectations)


# --- the scenario file ---

def test_the_shop_has_five_scenarios_from_its_file():
    assert list(shop_target.SCENARIOS) == [
        "refund_o1002",
        "refund_o1001",
        "stock_keyboard",
        "refund_o1005_again",
        "status_o1004_timeout",
    ]
    assert shop_target.SCENARIO_FILE.name == "shop.toml"


def test_every_shop_scenario_is_named_after_its_table_and_has_expectations():
    for name, scenario in shop_target.SCENARIOS.items():
        assert scenario.name == name
        assert scenario.expectations, f"{name} has no expectations for the judge"


def test_exactly_one_shop_scenario_times_out_on_purpose():
    timeouts = {
        name: scenario.timeout_seconds
        for name, scenario in shop_target.SCENARIOS.items()
        if scenario.timeout_seconds is not None
    }

    assert timeouts == {"status_o1004_timeout": 1}


def test_each_shop_scenario_is_about_the_order_or_product_it_names():
    """A scenario that mentions an id in its name must mention it to the customer, too."""
    scenarios = shop_target.SCENARIOS
    assert "O1001" in scenarios["refund_o1001"].persona and "O1001" in scenarios["refund_o1001"].goal
    assert "O1005" in scenarios["refund_o1005_again"].persona
    assert "O1005" in scenarios["refund_o1005_again"].goal
    assert "O1004" in scenarios["status_o1004_timeout"].persona
    assert "P2" in scenarios["stock_keyboard"].persona
