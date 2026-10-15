import asyncio
import copy
from typing import Any, TypedDict, Dict

ERR_ORDER_NOT_FOUND = "order_not_found"
ERR_PRODUCT_NOT_FOUND = "product_not_found"
ERR_NOT_DELIVERED = "order_not_delivered"
ERR_REFUND_EXCEEDS_TOTAL = "refund_exceeds_total"
ERR_INVALID_AMOUNT = "invalid_amount"

class Product(TypedDict):
    """A catalog entry."""

    product_id: str
    name: str
    price: int  # cents
    stock: int


class OrderItem(TypedDict):
    """One product line within an order."""

    product_id: str
    quantity: int


class Refund(TypedDict):
    """A single refund recorded against an order."""

    amount: int  # cents


class Order(TypedDict):
    """A stored order."""

    id: str
    status: str
    items: list[OrderItem]
    total: int  # cents
    refunds: list[Refund]


# Tool result: {"ok": True, ...data} or {"ok": False, "error": {...}}.
Result = Dict[str, Any]


class Shop:
    """A fake shop.

    Every method returns a dict: {"ok": True, ...data} or
    {"ok": False, "error": {"code": str, "message": str}}. Nothing raises for
    business-rule failures. Money is integer cents.
    """

    def __init__(self,
                 products: Dict[str, Product],
                 orders: Dict[str, Order]) -> None:
        self._products = copy.deepcopy(products)
        self._orders = copy.deepcopy(orders)
        self._lock = asyncio.Lock()

    @staticmethod
    def _error(code: str, message: str) -> Result:
        """Build a structured failure result.

        Args:
            code: Machine-readable error code (one of the ERR_* constants).
            message: Human-readable explanation of the failure.

        Returns:
            {"ok": False, "error": {"code": ..., "message": ...}}
        """
        return {"ok": False, "error": {"code": code, "message": message}}

    @staticmethod
    def _refunded(order: Order) -> int:
        """Return the total amount (cents) already refunded for an order.

        Args:
            order: The order whose refunds are summed.
        """
        return sum(r["amount"] for r in order["refunds"])

    def _view(self, order: Order) -> Dict[str, Any]:
        """Return a copy of an order with computed `refunded` and `refundable` fields.

        Args:
            order: The stored order. It is not modified.
        """
        view = copy.deepcopy(order)
        view["refunded"] = self._refunded(order)
        view["refundable"] = order["total"] - view["refunded"]
        return view

    async def get_order(self, order_id: str) -> Result:
        """Look up an order, including refunded and refundable amounts.

        Args:
            order_id: Id of the order, e.g. "O1001".

        Returns:
            {"ok": True, "order": ...} or an `order_not_found` error result.
        """
        order = self._orders.get(order_id)
        if order is None:
            return self._error(ERR_ORDER_NOT_FOUND, f"Order '{order_id}' does not exist.")
        return {"ok": True, "order": self._view(order)}

    async def check_stock(self, product_id: str) -> Result:
        """Report stock level and availability for a product.

        Args:
            product_id: Id of the product, e.g. "P1".

        Returns:
            {"ok": True, "product_id", "name", "stock", "in_stock"} or a
            `product_not_found` error result.
        """
        product = self._products.get(product_id)
        if product is None:
            return self._error(ERR_PRODUCT_NOT_FOUND, f"Product '{product_id}' does not exist.")
        return {
            "ok": True,
            "product_id": product_id,
            "name": product["name"],
            "stock": product["stock"],
            "in_stock": product["stock"] > 0,
        }

    async def refund_order(self, order_id: str, amount: int | None = None) -> Result:
        """Refund part or all of an order.

        The order must exist and be delivered, and cumulative refunds may not
        exceed the order total.

        Args:
            order_id: Id of the order to refund.
            amount: Amount to refund in cents. None refunds the whole remaining
                balance.

        Returns:
            On success, {"ok": True, "order_id", "refunded_amount",
            "total_refunded", "remaining_refundable"}. On failure, an error result
            with code `order_not_found`, `order_not_delivered`, `invalid_amount`
            or `refund_exceeds_total`.
        """
        async with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return self._error(ERR_ORDER_NOT_FOUND, f"Order '{order_id}' does not exist.")
            if order["status"] != "delivered":
                return self._error(
                    ERR_NOT_DELIVERED,
                    f"Order '{order_id}' is '{order['status']}'; only delivered orders can be refunded.",
                )
            remaining = order["total"] - self._refunded(order)
            if amount is None:
                if remaining == 0:
                    return self._error(
                        ERR_REFUND_EXCEEDS_TOTAL,
                        f"Order '{order_id}' is already fully refunded.",
                    )
                amount = remaining
            if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                return self._error(ERR_INVALID_AMOUNT, "Refund amount must be a positive integer (cents).")
            if amount > remaining:
                return self._error(
                    ERR_REFUND_EXCEEDS_TOTAL,
                    f"Refund of {amount} exceeds refundable balance of {remaining} for order '{order_id}'.",
                )
            order["refunds"].append({"amount": amount})
            return {
                "ok": True,
                "order_id": order_id,
                "refunded_amount": amount,
                "total_refunded": order["total"] - remaining + amount,
                "remaining_refundable": remaining - amount,
            }
