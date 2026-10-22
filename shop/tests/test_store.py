import asyncio

import pytest

from shop import store

pytestmark = pytest.mark.anyio


def assert_error(result, code):
    """Assert `result` is a structured failure with the given error code."""
    assert result["ok"] is False
    assert result["error"]["code"] == code
    assert result["error"]["message"]


# --- normal cases ---

async def test_get_order(shop):
    """An existing delivered order is returned with its refundable balance."""
    result = await shop.get_order("O1001")
    assert result["ok"] is True
    assert result["order"]["status"] == "delivered"
    assert result["order"]["refundable"] == 5000


async def test_check_stock_in_and_out(shop):
    """Stock check reports in-stock and out-of-stock products correctly."""
    assert (await shop.check_stock("P1"))["in_stock"] is True
    out = await shop.check_stock("P2")
    assert out["ok"] is True and out["in_stock"] is False and out["stock"] == 0


async def test_unknown_product(shop):
    """Stock check on an unknown product returns product_not_found."""
    assert_error(await shop.check_stock("NOPE"), store.ERR_PRODUCT_NOT_FOUND)


async def test_full_refund(shop):
    """Refund without an amount returns the entire order total."""
    result = await shop.refund_order("O1001")
    assert result["ok"] is True
    assert result["refunded_amount"] == 5000
    assert result["remaining_refundable"] == 0


async def test_partial_refund_then_rest(shop):
    """A partial refund followed by the remaining balance succeeds."""
    first = await shop.refund_order("O1001", 2000)
    assert first["ok"] and first["remaining_refundable"] == 3000
    second = await shop.refund_order("O1001", 3000)
    assert second["ok"] and second["total_refunded"] == 5000


async def test_invalid_amount(shop):
    """Zero and negative refund amounts are rejected as invalid_amount."""
    assert_error(await shop.refund_order("O1001", 0), store.ERR_INVALID_AMOUNT)
    assert_error(await shop.refund_order("O1001", -5), store.ERR_INVALID_AMOUNT)


# --- required failure cases ---

async def test_refund_unknown_order(shop):
    """Refunding an unknown order returns order_not_found."""
    assert_error(await shop.refund_order("O9999"), store.ERR_ORDER_NOT_FOUND)


async def test_get_unknown_order(shop):
    """Getting an unknown order returns order_not_found."""
    assert_error(await shop.get_order("O9999"), store.ERR_ORDER_NOT_FOUND)


async def test_refund_too_large(shop):
    """A refund above the order total is rejected and nothing is recorded."""
    assert_error(await shop.refund_order("O1001", 5001), store.ERR_REFUND_EXCEEDS_TOTAL)
    # nothing recorded
    assert (await shop.get_order("O1001"))["order"]["refunded"] == 0


async def test_double_refund(shop):
    """Refunding an already fully refunded order is rejected."""
    assert (await shop.refund_order("O1001"))["ok"] is True
    assert_error(await shop.refund_order("O1001"), store.ERR_REFUND_EXCEEDS_TOTAL)
    assert_error(await shop.refund_order("O1001", 1), store.ERR_REFUND_EXCEEDS_TOTAL)


async def test_refund_undelivered_order(shop):
    """Shipped and processing orders cannot be refunded."""
    assert_error(await shop.refund_order("O1002"), store.ERR_NOT_DELIVERED)  # shipped
    assert_error(await shop.refund_order("O1004"), store.ERR_NOT_DELIVERED)  # processing


async def test_seeded_partial_refund_respected(shop):
    """Refunds already on a seeded order count toward the total cap."""
    # O1005 already has 900 of 2700 refunded
    assert_error(await shop.refund_order("O1005", 2000), store.ERR_REFUND_EXCEEDS_TOTAL)
    assert (await shop.refund_order("O1005"))["refunded_amount"] == 1800


# --- concurrent refunds ---

async def test_concurrent_full_refunds_pay_once(shop):
    """Two simultaneous full refunds of one order: one succeeds, one is
    rejected.

    NOTE: Test pass without the lock today and fail if an await is added and
    the lock is removed.
    """
    results = await asyncio.gather(
        shop.refund_order("O1001"),
        shop.refund_order("O1001"),
    )

    succeeded = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    assert len(succeeded) == 1 and succeeded[0]["refunded_amount"] == 5000
    assert len(failed) == 1
    assert_error(failed[0], store.ERR_REFUND_EXCEEDS_TOTAL)
    assert (await shop.get_order("O1001"))["order"]["refunded"] == 5000


async def test_concurrent_partial_refunds_respect_the_cap(shop):
    """Two simultaneous 3000 refunds of a 5000 order: the second would exceed
    the total.

    NOTE: Test pass without the lock today and fail if an await is added and
    the lock is removed.
    """
    results = await asyncio.gather(
        shop.refund_order("O1001", 3000),
        shop.refund_order("O1001", 3000),
    )

    assert sorted(r["ok"] for r in results) == [False, True]
    assert (await shop.get_order("O1001"))["order"]["refunded"] == 3000


# --- one shop per conversation ---

ORDERS = {
    "O1": {
        "id": "O1",
        "status": "delivered",
        "items": [{"product_id": "P1", "quantity": 1}],
        "total": 1000,
        "refunds": [],
    },
}


async def test_shops_do_not_share_state():
    """Two shops built from the same data do not see each other's refunds."""
    first, second = store.Shop({}, ORDERS), store.Shop({}, ORDERS)

    assert (await first.refund_order("O1", 400))["ok"] is True

    assert (await first.get_order("O1"))["order"]["refunded"] == 400
    assert (await second.get_order("O1"))["order"]["refunded"] == 0
    assert (await second.refund_order("O1"))["refunded_amount"] == 1000


async def test_shop_does_not_change_the_data_it_was_built_from():
    """The shop works on its own copy, so the data can build another shop later."""
    shop = store.Shop({}, ORDERS)

    await shop.refund_order("O1")

    assert ORDERS["O1"]["refunds"] == []
