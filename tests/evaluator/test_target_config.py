"""Reading a target from a TOML file, and opening it."""

import json
from pathlib import Path

import httpx
import pytest

from evaluator.http_target import DEFAULT_BODY_TEMPLATE
from evaluator.target import ToolCallSource
from evaluator.target_config import Auth, TargetConfigError, load_target_config

SECRET = "s3cr3t-value-123"

FULL = """
name = "acme-support"
base_url = "https://agent.example.com/api"
path = "/v1/chat"
request_timeout_seconds = 12.5

[auth]
header = "Authorization"
env = "ACME_TOKEN"
prefix = "Bearer "

[headers]
X-Client = "mystery-shopper"

[request]
body = { session = "{conversation_id}", input = { text = "{message}", lang = "en" }, stream = false }

[response]
reply = "data.messages.0.text"
"""


def write(tmp_path: Path, text: str, name: str = "acme.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- reading the file ---

def test_a_file_with_only_the_base_url_gets_the_defaults(tmp_path):
    """The defaults are the shop's shape, so a target like it needs one line."""
    config = load_target_config(write(tmp_path, 'base_url = "http://127.0.0.1:8000"', name="local-shop.toml"))

    assert config.name == "local-shop"
    assert config.base_url == "http://127.0.0.1:8000"
    assert config.path == "/chat"
    assert config.request_timeout_seconds == 60.0
    assert config.auth is None
    assert dict(config.headers) == {}
    assert config.body_template == DEFAULT_BODY_TEMPLATE
    assert config.reply_path == ("reply",)


def test_every_key_of_a_full_file_is_read(tmp_path):
    path = write(tmp_path, FULL)

    config = load_target_config(path)

    assert config.name == "acme-support"
    assert config.source == path
    assert config.base_url == "https://agent.example.com/api"
    assert config.path == "/v1/chat"
    assert config.request_timeout_seconds == 12.5
    assert config.auth == Auth(header="Authorization", env="ACME_TOKEN", prefix="Bearer ")
    assert dict(config.headers) == {"X-Client": "mystery-shopper"}
    assert config.body_template == {
        "session": "{conversation_id}", "input": {"text": "{message}", "lang": "en"}, "stream": False,
    }
    assert config.reply_path == ("data", "messages", "0", "text")


def test_a_whole_number_timeout_is_taken_as_seconds(tmp_path):
    config = load_target_config(write(tmp_path, 'base_url = "https://a.example"\nrequest_timeout_seconds = 30'))

    assert config.request_timeout_seconds == 30.0


def test_a_marker_may_be_used_more_than_once_and_in_a_list(tmp_path):
    text = '''base_url = "https://a.example"
[request]
body = { messages = [{ role = "user", content = "{message}" }], echo = "{message}" }
'''
    config = load_target_config(write(tmp_path, text))

    assert config.body_template["messages"] == [{"role": "user", "content": "{message}"}]


def test_a_body_without_the_conversation_id_is_allowed(tmp_path):
    """Some endpoints keep the conversation by another means; the message is what must be sent."""
    text = 'base_url = "https://a.example"\n[request]\nbody = { text = "{message}" }\n'

    assert load_target_config(write(tmp_path, text)).body_template == {"text": "{message}"}


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8000", "http://127.0.0.1:8000", "http://127.0.0.2", "http://[::1]:8000", "http://LOCALHOST"],
)
def test_the_auth_header_may_go_over_plain_http_to_this_machine(tmp_path, base_url):
    text = f'base_url = "{base_url}"\n[auth]\nheader = "X-Api-Key"\nenv = "ACME_TOKEN"\n'

    assert load_target_config(write(tmp_path, text)).base_url == base_url


def test_without_auth_plain_http_to_another_machine_is_allowed(tmp_path):
    """Nothing secret is sent, so there is nothing to protect."""
    assert load_target_config(write(tmp_path, 'base_url = "http://agent.example.com"')).auth is None


# --- a file that is wrong ---

@pytest.mark.parametrize(
    "text, message",
    [
        ('path = "/chat"', "the key 'base_url' is missing"),
        ('base_url = "https://a.example"\nbase-url = "x"', "unknown key(s): base-url"),
        ('base_url = "https://a.example"\n[auth]\nheader = "A"\nenv = "E"\nvalue = "x"', "unknown key(s) in [auth]: value"),
        ('base_url = "https://a.example"\n[request]\nbody = { m = "{message}" }\nmethod = "PUT"', "unknown key(s) in [request]: method"),
        ('base_url = "https://a.example"\n[response]\nreply = "r"\nstatus = "s"', "unknown key(s) in [response]: status"),
        ('base_url = 5', "'base_url' must be text"),
        ('base_url = "ftp://a.example"', "must start with http:// or https://"),
        ('base_url = "https://"', "and name a host"),
        ('base_url = "a.example"', "must start with http:// or https://"),
        ('base_url = "https://user:pw@a.example"', "must not hold a user name or password"),
        ('base_url = "https://a.example?key=abc"', "must not have a query or a fragment"),
        ('base_url = "https://a.example#x"', "must not have a query or a fragment"),
        ('base_url = "https://a.example"\nname = "my agent"', "'name' must be letters, digits"),
        ('base_url = "https://a.example"\nname = 3', "'name' must be letters, digits"),
        ('base_url = "https://a.example"\npath = "chat"', "'path' must be text that starts with '/'"),
        ('base_url = "https://a.example"\nrequest_timeout_seconds = 0', "greater than 0"),
        ('base_url = "https://a.example"\nrequest_timeout_seconds = -1', "greater than 0"),
        ('base_url = "https://a.example"\nrequest_timeout_seconds = true', "greater than 0"),
        ('base_url = "https://a.example"\nrequest_timeout_seconds = inf', "greater than 0"),
        ('base_url = "https://a.example"\nrequest_timeout_seconds = nan', "greater than 0"),
        ('base_url = "https://a.example"\nauth = "token"', "'auth' must be a table"),
        ('base_url = "https://a.example"\n[auth]\nenv = "E"', "[auth] 'header' must be the name of an HTTP header"),
        ('base_url = "https://a.example"\n[auth]\nheader = "Bad Header"\nenv = "E"', "[auth] 'header' must be"),
        ('base_url = "https://a.example"\n[auth]\nheader = "A"', "[auth] 'env' must be the name of an environment variable"),
        ('base_url = "https://a.example"\n[auth]\nheader = "A"\nenv = "1BAD"', "[auth] 'env' must be"),
        ('base_url = "https://a.example"\n[auth]\nheader = "A"\nenv = "E"\nprefix = 1', "[auth] 'prefix' must be text"),
        ('base_url = "http://agent.example.com"\n[auth]\nheader = "A"\nenv = "E"', "plain http to another machine"),
        ('base_url = "http://10.0.0.5"\n[auth]\nheader = "A"\nenv = "E"', "plain http to another machine"),
        ('base_url = "https://a.example"\nheaders = "x"', "'headers' must be a table"),
        ('base_url = "https://a.example"\n[headers]\n"Bad Name" = "x"', "is not the name of an HTTP header"),
        ('base_url = "https://a.example"\n[headers]\nX-Count = 3', "'X-Count' must be text"),
        ('base_url = "https://a.example"\n[headers]\nAuthorization = "Bearer abc"', "carries a secret"),
        ('base_url = "https://a.example"\n[headers]\nproxy-authorization = "x"', "carries a secret"),
        ('base_url = "https://a.example"\n[auth]\nheader = "X-Api-Key"\nenv = "E"\n[headers]\nx-api-key = "v"',
         "is also the [auth] header"),
        ('base_url = "https://a.example"\nrequest = 1', "'request' must be a table"),
        ('base_url = "https://a.example"\n[request]', "[request] the key 'body' is missing"),
        ('base_url = "https://a.example"\n[request]\nbody = "{message}"', "[request] 'body' must be a table"),
        ('base_url = "https://a.example"\n[request]\nbody = { id = "{conversation_id}" }', "must have the value \"{message}\""),
        ('base_url = "https://a.example"\n[request]\nbody = { text = "Customer says: {message}" }',
         "request.body.text: 'Customer says: {message}' has {message} inside a longer text"),
        ('base_url = "https://a.example"\n[request]\nbody = { m = "{message}", s = ["x", "shopper-{conversation_id}"] }',
         "request.body.s.1: 'shopper-{conversation_id}' has {conversation_id} inside a longer text"),
        ('base_url = "https://a.example"\n[request]\nbody = { m = "{message}", when = 2026-11-27 }',
         "request.body.when: a date cannot be sent as JSON"),
        ('base_url = "https://a.example"\nresponse = 1', "'response' must be a table"),
        ('base_url = "https://a.example"\n[response]', "[response] 'reply' must be the place of the reply"),
        ('base_url = "https://a.example"\n[response]\nreply = ""', "[response] 'reply' must be"),
        ('base_url = "https://a.example"\n[response]\nreply = "data..text"', "[response] 'reply' must be"),
        ('base_url = "https://a.example"\n[response]\nreply = 3', "[response] 'reply' must be"),
    ],
)
def test_a_wrong_file_is_refused_with_a_message_that_names_the_file_and_the_problem(tmp_path, text, message):
    path = write(tmp_path, text)

    with pytest.raises(TargetConfigError) as error:
        load_target_config(path)

    assert str(error.value).startswith(f"{path}: ")
    assert message in str(error.value)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(TargetConfigError, match="Cannot read the target config"):
        load_target_config(tmp_path / "nowhere.toml")


def test_a_file_that_is_not_toml_is_refused(tmp_path):
    with pytest.raises(TargetConfigError, match="is not valid TOML"):
        load_target_config(write(tmp_path, "base_url = "))


# --- the secret ---

def test_a_missing_secret_is_found_before_any_target_is_opened(tmp_path, monkeypatch):
    monkeypatch.delenv("ACME_TOKEN", raising=False)
    config = load_target_config(write(tmp_path, FULL))

    with pytest.raises(TargetConfigError, match="Environment variable ACME_TOKEN is not set"):
        config.check_secret()


def test_a_blank_secret_counts_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_TOKEN", "   ")
    config = load_target_config(write(tmp_path, FULL))

    with pytest.raises(TargetConfigError, match="is not set"):
        config.check_secret()


def test_a_secret_of_more_than_one_line_is_refused_without_showing_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_TOKEN", "abc\ndef")
    config = load_target_config(write(tmp_path, FULL))

    with pytest.raises(TargetConfigError) as error:
        config.check_secret()

    assert "more than one line" in str(error.value)
    assert "abc" not in str(error.value) and "def" not in str(error.value)


def test_without_auth_no_secret_is_needed(tmp_path):
    config = load_target_config(write(tmp_path, 'base_url = "https://a.example"'))

    config.check_secret()
    assert config.request_headers() == {}


def test_the_secret_goes_in_the_auth_header_after_the_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_TOKEN", f"  {SECRET}  ")
    config = load_target_config(write(tmp_path, FULL))

    config.check_secret()
    assert config.request_headers() == {"X-Client": "mystery-shopper", "Authorization": f"Bearer {SECRET}"}


def test_the_config_holds_no_secret(tmp_path, monkeypatch):
    """The secret is read when it is needed, so printing the config can never show it."""
    monkeypatch.setenv("ACME_TOKEN", SECRET)
    config = load_target_config(write(tmp_path, FULL))

    assert SECRET not in repr(config)
    assert "ACME_TOKEN" in repr(config)


# --- opening a target ---

@pytest.mark.anyio
async def test_an_opened_target_sends_the_request_the_file_describes(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_TOKEN", SECRET)
    config = load_target_config(write(tmp_path, FULL))
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": {"messages": [{"text": "Hello from Acme"}]}})

    async with config.open_target(transport=httpx.MockTransport(handler)) as target:
        reply = await target.send("Where is my order?")
        conversation_id = target.conversation_id

    request = requests[0]
    assert reply == "Hello from Acme"
    assert request.method == "POST"
    assert str(request.url) == "https://agent.example.com/api/v1/chat"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    assert request.headers["X-Client"] == "mystery-shopper"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "session": conversation_id, "input": {"text": "Where is my order?", "lang": "en"}, "stream": False,
    }
    assert request.extensions["timeout"] == {"connect": 12.5, "read": 12.5, "write": 12.5, "pool": 12.5}


@pytest.mark.anyio
async def test_every_opened_target_is_a_conversation_of_its_own(tmp_path):
    config = load_target_config(write(tmp_path, 'base_url = "https://a.example"'))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"reply": "ok"}))

    async with config.open_target(transport=transport) as first, config.open_target(transport=transport) as second:
        assert first.conversation_id != second.conversation_id


@pytest.mark.anyio
async def test_an_opened_target_keeps_no_tool_log(tmp_path):
    """A black box: the results of its runs have no `tool_calls`, which means "cannot tell"."""
    config = load_target_config(write(tmp_path, 'base_url = "https://a.example"'))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"reply": "ok"}))

    async with config.open_target(transport=transport) as target:
        assert not isinstance(target, ToolCallSource)


@pytest.mark.anyio
async def test_a_target_is_not_opened_without_its_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("ACME_TOKEN", raising=False)
    config = load_target_config(write(tmp_path, FULL))

    with pytest.raises(TargetConfigError, match="ACME_TOKEN is not set"):
        async with config.open_target(transport=httpx.MockTransport(lambda request: httpx.Response(200))):
            pass
