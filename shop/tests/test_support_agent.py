import pytest

from shop import support_agent

pytestmark = pytest.mark.anyio

TOOL_NAMES = {"get_order", "check_stock", "refund_order"}


def test_tools_for_exposes_only_the_support_tools(shop):
    """Methods of the shop that are not support tools, like `_view`, cannot be called by the agent."""
    assert set(support_agent.tools_for(shop)) == TOOL_NAMES


def test_every_tool_has_a_schema_and_the_other_way_round(shop):
    """The model is offered exactly the tools that can run."""
    offered = {schema["name"] for schema in support_agent.SPEC.tool_schemas}

    assert offered == set(support_agent.tools_for(shop))


async def test_tools_act_on_the_shop_they_were_built_for(shop):
    """A tool table is bound to its own shop, not to a shared one."""
    other = type(shop)(products={}, orders={})

    tools = support_agent.tools_for(shop)
    refund = await tools["refund_order"](order_id="O1001")

    assert refund["ok"] is True
    assert (await shop.get_order("O1001"))["order"]["refunded"] == 5000
    assert (await other.get_order("O1001"))["ok"] is False
