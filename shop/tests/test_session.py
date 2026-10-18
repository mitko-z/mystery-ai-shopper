import pytest

from shop.session import EmptyReplyError, SupportSession
from tests.fakes import ScriptedInteractions, call_step, reply_step, use_fake, user_step

pytestmark = pytest.mark.anyio


async def test_send_returns_the_final_text(monkeypatch):
    """The reply is the agent's final text, not the thoughts and tool steps before it."""
    fake = ScriptedInteractions(
        [call_step("c1", "check_stock", {"product_id": "P1"})],
        [reply_step("In stock. "), reply_step("12 left.")],
    )
    use_fake(monkeypatch, fake)

    reply = await SupportSession().send("stock of P1?")

    assert reply == "In stock. 12 left."


async def test_send_keeps_history_and_the_tool_call_log(monkeypatch):
    """Each message continues the same history, and every tool call lands in the log."""
    fake = ScriptedInteractions(
        [call_step("c1", "refund_order", {"order_id": "O1001"})],
        [reply_step("Refunded 50.00.")],
        [reply_step("Anything else?")],
    )
    use_fake(monkeypatch, fake)
    session = SupportSession()

    await session.send("refund O1001")
    await session.send("thanks")

    assert [call.name for call in session.tool_calls] == ["refund_order"]
    assert fake.calls[2]["input"][0] == user_step("refund O1001")
    assert (await session.shop.get_order("O1001"))["order"]["refunded"] == 5000


async def test_sessions_do_not_share_a_shop():
    """A new session starts from the seed shop, whatever other sessions did."""
    first, second = SupportSession(), SupportSession()

    await first.shop.refund_order("O1001")

    assert first.shop is not second.shop
    assert (await second.shop.get_order("O1001"))["order"]["refunded"] == 0


@pytest.mark.parametrize(
    "final_step",
    [
        {"type": "model_output"},
        {"type": "model_output", "content": []},
        {"type": "model_output", "content": [{"type": "text", "text": "  "}]},
    ],
    ids=["no-content", "no-parts", "blank-text"],
)
async def test_final_answer_without_text_is_an_error(monkeypatch, final_step):
    """A blank final answer raises, and the session stays usable."""
    use_fake(monkeypatch, ScriptedInteractions([final_step], [reply_step("Sorry, hello.")]))
    session = SupportSession()

    with pytest.raises(EmptyReplyError):
        await session.send("hi")

    assert await session.send("hi again") == "Sorry, hello."
