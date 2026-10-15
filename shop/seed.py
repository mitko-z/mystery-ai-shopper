from shop import store


def get_seed_shop() -> store.Shop:
    """Return a Shop instance with the seed data.

    Returns:
        A Shop instance with the seed data.
    """
    products = {
        "P1": store.Product(
            product_id="P1",
            name="Wireless Mouse",
            price=2500,
            stock=12
        ),
        "P2": store.Product(
            product_id="P2",
            name="Mechanical Keyboard",
            price=8900,
            stock=0
        ),
        "P3": store.Product(
            product_id="P3",
            name="USB-C Cable",
            price=900,
            stock=150
        ),
        "P4": store.Product(
            product_id="P4",
            name="27in Monitor",
            price=21000,
            stock=3
        ),
        "P5": store.Product(
            product_id="P5",
            name="Laptop Stand",
            price=3500,
            stock=0
        ),
    }
    orders = {
        "O1001": store.Order(
            id="O1001",
            status="delivered",
            items=[{"product_id": "P1", "quantity": 2}],
            total=5000,
            refunds=[]
        ),
        "O1002": store.Order(
            id="O1002",
            status="shipped",
            items=[{"product_id": "P4", "quantity": 1}],
            total=21000,
            refunds=[]
        ),
        "O1003": store.Order(
            id="O1003",
            status="delivered",
            items=[{"product_id": "P2", "quantity": 1}, {"product_id": "P3", "quantity": 1}],
            total=9800,
            refunds=[]
        ),
        "O1004": store.Order(
            id="O1004",
            status="processing",
            items=[{"product_id": "P5", "quantity": 1}],
            total=3500,
            refunds=[]
        ),
        "O1005": store.Order(
            id="O1005",
            status="delivered",
            items=[{"product_id": "P3", "quantity": 3}],
            total=2700,
            refunds=[{"amount": 900}]
        ),
    }
    return store.Shop(products=products, orders=orders)
