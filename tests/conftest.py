import pytest

from shop import seed, store


@pytest.fixture
def shop() -> store.Shop:
    """A new shop with the seed data, so a test never sees another test's refunds."""
    return seed.get_seed_shop()
