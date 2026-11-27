"""The example target config `target_configs/shop_local.toml`, tried on the shop's own HTTP API."""

from pathlib import Path

import httpx
import pytest

from evaluator.target import ToolCallSource
from evaluator.target_config import load_target_config
from shop import app as shop_app
from tests.fakes import ScriptedInteractions, reply_step, use_fake, user_step

pytestmark = pytest.mark.anyio

EXAMPLE = Path(__file__).resolve().parents[2] / "target_configs" / "shop_local.toml"


@pytest.fixture(autouse=True)
def fresh_state():
    """Forget the shop's stored conversations before every test."""
    shop_app._conversations.clear()


def test_the_example_config_points_at_the_local_server():
    config = load_target_config(EXAMPLE)

    assert config.name == "shop-local"
    assert config.base_url == "http://127.0.0.1:8000"
    assert config.auth is None


async def test_the_example_config_speaks_the_shops_api(monkeypatch):
    """The shop app answers in the same process, as the server would, so the request and reply shapes are checked."""
    fake = ScriptedInteractions([reply_step("Hello! How can I help?")], [reply_step("Sure.")])
    use_fake(monkeypatch, fake)
    config = load_target_config(EXAMPLE)

    async with config.open_target(transport=httpx.ASGITransport(app=shop_app.app)) as target:
        first = await target.send("hi")
        second = await target.send("and O1001?")
        conversation_id = target.conversation_id

    assert (first, second) == ("Hello! How can I help?", "Sure.")
    assert fake.calls[1]["input"][0] == user_step("hi")  # one conversation, kept by its id
    assert conversation_id in shop_app._conversations
    assert not isinstance(target, ToolCallSource)
