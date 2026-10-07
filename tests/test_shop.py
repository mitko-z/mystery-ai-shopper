import pytest

from shop import store

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    """Run anyio-marked tests on asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset in-memory data before every test."""
    store.reset()


def assert_error(result, code):
    """Assert `result` is a structured failure with the given error code."""
    assert result["ok"] is False
    assert result["error"]["code"] == code
    assert result["error"]["message"]


# --- normal cases ---

async def test_get_order():
    """An existing delivered order is returned with its refundable balance."""
    result = await store.get_order("O1001")
    assert result["ok"] is True
    assert result["order"]["status"] == "delivered"
    assert result["order"]["refundable"] == 5000


async def test_check_stock_in_and_out():
    """Stock check reports in-stock and out-of-stock products correctly."""
    assert (await store.check_stock("P1"))["in_stock"] is True
    out = await store.check_stock("P2")
    assert out["ok"] is True and out["in_stock"] is False and out["stock"] == 0


async def test_unknown_product():
    """Stock check on an unknown product returns product_not_found."""
    assert_error(await store.check_stock("NOPE"), store.ERR_PRODUCT_NOT_FOUND)


async def test_full_refund():
    """Refund without an amount returns the entire order total."""
    result = await store.refund_order("O1001")
    assert result["ok"] is True
    assert result["refunded_amount"] == 5000
    assert result["remaining_refundable"] == 0


async def test_partial_refund_then_rest():
    """A partial refund followed by the remaining balance succeeds."""
    first = await store.refund_order("O1001", 2000)
    assert first["ok"] and first["remaining_refundable"] == 3000
    second = await store.refund_order("O1001", 3000)
    assert second["ok"] and second["total_refunded"] == 5000


async def test_invalid_amount():
    """Zero and negative refund amounts are rejected as invalid_amount."""
    assert_error(await store.refund_order("O1001", 0), store.ERR_INVALID_AMOUNT)
    assert_error(await store.refund_order("O1001", -5), store.ERR_INVALID_AMOUNT)


# --- required failure cases ---

async def test_refund_unknown_order():
    """Refunding an unknown order returns order_not_found."""
    assert_error(await store.refund_order("O9999"), store.ERR_ORDER_NOT_FOUND)


async def test_get_unknown_order():
    """Getting an unknown order returns order_not_found."""
    assert_error(await store.get_order("O9999"), store.ERR_ORDER_NOT_FOUND)


async def test_refund_too_large():
    """A refund above the order total is rejected and nothing is recorded."""
    assert_error(await store.refund_order("O1001", 5001), store.ERR_REFUND_EXCEEDS_TOTAL)
    # nothing recorded
    assert (await store.get_order("O1001"))["order"]["refunded"] == 0


async def test_double_refund():
    """Refunding an already fully refunded order is rejected."""
    assert (await store.refund_order("O1001"))["ok"] is True
    assert_error(await store.refund_order("O1001"), store.ERR_REFUND_EXCEEDS_TOTAL)
    assert_error(await store.refund_order("O1001", 1), store.ERR_REFUND_EXCEEDS_TOTAL)


async def test_refund_undelivered_order():
    """Shipped and processing orders cannot be refunded."""
    assert_error(await store.refund_order("O1002"), store.ERR_NOT_DELIVERED)  # shipped
    assert_error(await store.refund_order("O1004"), store.ERR_NOT_DELIVERED)  # processing


async def test_seeded_partial_refund_respected():
    """Refunds already on a seeded order count toward the total cap."""
    # O1005 already has 900 of 2700 refunded
    assert_error(await store.refund_order("O1005", 2000), store.ERR_REFUND_EXCEEDS_TOTAL)
    assert (await store.refund_order("O1005"))["refunded_amount"] == 1800

