import pytest

from evaluator.customer import FINISH_SCHEMA, KICKOFF, CustomerError, SimulatedCustomer
from evaluator.outcome import Outcome, OutcomeKind
from evaluator.scenario import Scenario
from tests.fakes import ScriptedInteractions, call_step, reply_step, use_fake, user_step

pytestmark = pytest.mark.anyio

SCENARIO = Scenario(
    name="refund",
    persona="You are Alex and you bought a monitor, order O1002.",
    goal="Get a refund for order O1002.",
)


async def test_first_turn_starts_from_the_kickoff(monkeypatch):
    """With nothing from the target yet, the customer is told to write its first message."""
    fake = ScriptedInteractions([reply_step("Hi, where is my order O1002?")])
    use_fake(monkeypatch, fake)
    customer = SimulatedCustomer(SCENARIO)

    text = await customer.next_message(None)

    assert text == "Hi, where is my order O1002?"
    assert fake.calls[0]["input"] == [user_step(KICKOFF)]
    assert customer.outcome is None


async def test_the_model_is_told_who_to_play_and_may_only_finish(monkeypatch):
    """The persona and the goal are in the instruction, and `finish` is the only tool."""
    fake = ScriptedInteractions([reply_step("Hi")])
    use_fake(monkeypatch, fake)

    await SimulatedCustomer(SCENARIO).next_message(None)

    sent = fake.calls[0]
    assert SCENARIO.persona in sent["system_instruction"]
    assert SCENARIO.goal in sent["system_instruction"]
    assert sent["tools"] == [FINISH_SCHEMA]


def test_the_customer_cannot_choose_cut_off():
    """`cut_off` is decided by the runner, so the model is not offered it."""
    offered = FINISH_SCHEMA["parameters"]["properties"]["outcome"]["enum"]

    assert offered == ["goal_achieved", "gave_up"]


async def test_the_targets_reply_is_the_next_input(monkeypatch):
    """The customer sees the conversation from its side: its own words, then what the target said."""
    fake = ScriptedInteractions([reply_step("Hi, refund please.")], [reply_step("It is shipped, why?")])
    use_fake(monkeypatch, fake)
    customer = SimulatedCustomer(SCENARIO)

    await customer.next_message(None)
    text = await customer.next_message("Sorry, that order is still shipped.")

    assert text == "It is shipped, why?"
    assert fake.calls[1]["input"] == [
        user_step(KICKOFF),
        reply_step("Hi, refund please."),
        user_step("Sorry, that order is still shipped."),
    ]


async def test_finish_sets_the_outcome_and_returns_the_goodbye(monkeypatch):
    """After `finish` the customer's last text is its goodbye, and the outcome is set."""
    fake = ScriptedInteractions(
        [call_step("f1", "finish", {"outcome": "gave_up", "reason": "The refund was refused."})],
        [reply_step("Okay, bye.")],
    )
    use_fake(monkeypatch, fake)
    customer = SimulatedCustomer(SCENARIO)

    text = await customer.next_message("No refund is possible.")

    assert text == "Okay, bye."
    assert customer.outcome == Outcome(OutcomeKind.GAVE_UP, "The refund was refused.")


@pytest.mark.parametrize("outcome", ["cut_off", "quit"], ids=["cut-off", "made-up"])
async def test_finish_with_an_outcome_the_customer_may_not_choose_is_refused(monkeypatch, outcome):
    """The model is told which outcomes are allowed, and the conversation goes on."""
    fake = ScriptedInteractions(
        [call_step("f1", "finish", {"outcome": outcome, "reason": "x"})],
        [reply_step("Let me think again.")],
    )
    use_fake(monkeypatch, fake)
    customer = SimulatedCustomer(SCENARIO)

    text = await customer.next_message("Hello")

    assert customer.outcome is None
    assert text == "Let me think again."
    result = fake.calls[1]["input"][-1]
    assert result["is_error"] is True
    assert "goal_achieved" in result["result"] and "gave_up" in result["result"]


async def test_an_empty_message_without_finish_is_an_error(monkeypatch):
    """Sending a blank message to the target would be pointless, so the customer fails loudly."""
    use_fake(monkeypatch, ScriptedInteractions([reply_step("   ")]))

    with pytest.raises(CustomerError):
        await SimulatedCustomer(SCENARIO).next_message(None)
