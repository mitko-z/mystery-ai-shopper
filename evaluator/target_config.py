"""Read a target from a TOML file: the HTTP chat endpoint of an agent.

A target of this kind needs no Python module in `targets/`. The file says where
the endpoint is, how a request looks and where the reply is in the answer:

    name = "acme-support"                 # optional, default: the file's name
    base_url = "https://agent.example.com"
    path = "/v1/chat"                     # optional, default "/chat"
    request_timeout_seconds = 60          # optional, default 60: one HTTP request

    [auth]                                # optional
    header = "Authorization"
    env = "ACME_TOKEN"                    # the secret is read from this variable
    prefix = "Bearer "                    # optional, put before the secret

    [headers]                             # optional, static and not secret
    X-Client = "mystery-shopper"

    [request]                             # optional, default: the shop's body
    body = { session = "{conversation_id}", input = { text = "{message}" } }

    [response]                            # optional
    reply = "data.messages.0.text"        # default "reply"

The secret is never in the file. It is read from the environment when a target
is opened, and it is never printed. A wrong file is refused with a message that
names the file and the key, before any run starts.
"""

import ipaddress
import math
import os
import re
import tomllib
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from evaluator.http_target import (
    DEFAULT_BODY_TEMPLATE,
    DEFAULT_REPLY_PATH,
    MARKERS,
    MESSAGE_MARKER,
    HttpChatTarget,
)
from evaluator.target import Target

DEFAULT_PATH = "/chat"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0

_TOP_KEYS = ("name", "base_url", "path", "request_timeout_seconds", "auth", "headers", "request", "response")
_AUTH_KEYS = ("header", "env", "prefix")
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_ENV_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The characters of an HTTP header name (a "token" in RFC 9110).
_HEADER_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
# Headers that carry a secret. They belong in [auth], which reads the secret from the environment.
_SECRET_HEADERS = frozenset({"authorization", "proxy-authorization"})


class TargetConfigError(Exception):
    """Raised when a target config cannot be read, is wrong, or its secret is not set."""


# Makes the error for a problem in the file, with the file's name in front.
_Fail = Callable[[str], TargetConfigError]


@dataclass(frozen=True)
class Auth:
    """Which header carries the secret, and where the secret is read from.

    Attributes:
        header: The name of the header, e.g. `Authorization`.
        env: The environment variable that holds the secret.
        prefix: Text put before the secret, e.g. `Bearer `.
    """

    header: str
    env: str
    prefix: str = ""


@dataclass(frozen=True)
class HttpTargetConfig:
    """An HTTP chat endpoint, as a target config file describes it.

    It has `open_target()` like a module of the `targets` package, so the
    runner uses it the same way. It holds no secret, only where to find it.

    Attributes:
        name: The target's name in the results, e.g. `acme-support`.
        source: The file it was read from.
        base_url: Where the endpoint is, e.g. `https://agent.example.com`.
        path: The chat endpoint.
        request_timeout_seconds: How long one HTTP request may take.
        headers: Static headers, sent with every request.
        auth: Where the secret comes from, or None if no secret is sent.
        body_template: The request body, with markers (see `fill_template`).
        reply_path: Where the reply is in the JSON answer.
    """

    name: str
    source: Path
    base_url: str
    path: str = DEFAULT_PATH
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    headers: Mapping[str, str] = field(default_factory=dict)
    auth: Optional[Auth] = None
    body_template: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_BODY_TEMPLATE))
    reply_path: tuple[str, ...] = DEFAULT_REPLY_PATH

    def check_secret(self) -> None:
        """Make sure the secret can be read, so a missing one is found before any run starts.

        Raises:
            TargetConfigError: If the variable is unset, blank or not one line.
        """
        if self.auth is not None:
            self._read_secret(self.auth)

    def _read_secret(self, auth: Auth) -> str:
        """The secret, from the environment. A message about it never holds its value."""
        secret = os.environ.get(auth.env, "").strip()
        if not secret:
            raise TargetConfigError(
                f"Environment variable {auth.env} is not set. The target config {self.source} "
                "reads the secret for its auth header from it."
            )
        if "\n" in secret or "\r" in secret:
            raise TargetConfigError(f"Environment variable {auth.env} holds more than one line.")
        return secret

    def request_headers(self) -> dict[str, str]:
        """All headers of a request: the static ones and the auth header with its secret.

        Raises:
            TargetConfigError: If the secret cannot be read.
        """
        headers = dict(self.headers)
        if self.auth is not None:
            headers[self.auth.header] = self.auth.prefix + self._read_secret(self.auth)
        return headers

    @asynccontextmanager
    async def open_target(
        self, transport: Optional[httpx.AsyncBaseTransport] = None
    ) -> AsyncGenerator[Target]:
        """Open a chat with the endpoint, in a conversation of its own.

        Args:
            transport: Where the requests go instead of the network, for tests.

        Yields:
            A target that sends each message to the endpoint. It keeps no tool
            log, so the results of its runs have no `tool_calls`.

        Raises:
            TargetConfigError: If the secret cannot be read.
        """
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.request_headers(),
            timeout=self.request_timeout_seconds,
            transport=transport,
        ) as client:
            yield HttpChatTarget(
                client,
                conversation_id=str(uuid.uuid4()),
                path=self.path,
                body_template=self.body_template,
                reply_path=self.reply_path,
            )


def load_target_config(path: str | Path) -> HttpTargetConfig:
    """Read and check a target config file.

    Args:
        path: The TOML file.

    Returns:
        The target it describes. The secret is not read yet (see `check_secret`).

    Raises:
        TargetConfigError: If the file cannot be read, is not valid TOML, or
            something in it is wrong. The message names the file and the key.
    """
    path = Path(path)
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except OSError as e:
        raise TargetConfigError(f"Cannot read the target config {path}: {e.strerror or e}") from e
    except tomllib.TOMLDecodeError as e:
        raise TargetConfigError(f"The target config {path} is not valid TOML: {e}") from e

    def fail(problem: str) -> TargetConfigError:
        return TargetConfigError(f"{path}: {problem}")

    _refuse_unknown_keys(data, _TOP_KEYS, "", fail)

    name = data.get("name", path.stem)
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise fail("'name' must be letters, digits, '-' or '_', because it is part of the results file's name.")

    auth = _read_auth(data.get("auth"), fail)
    base_url = _read_base_url(data, auth, fail)

    target_path = data.get("path", DEFAULT_PATH)
    if not isinstance(target_path, str) or not target_path.startswith("/"):
        raise fail("'path' must be text that starts with '/', e.g. \"/chat\".")

    timeout = data.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    if not _is_number(timeout) or not math.isfinite(timeout) or timeout <= 0:
        raise fail("'request_timeout_seconds' must be a number greater than 0.")

    return HttpTargetConfig(
        name=name,
        source=path,
        base_url=base_url,
        path=target_path,
        request_timeout_seconds=float(timeout),
        headers=_read_headers(data.get("headers", {}), auth, fail),
        auth=auth,
        body_template=_read_body(data.get("request"), fail),
        reply_path=_read_reply_path(data.get("response"), fail),
    )


def _refuse_unknown_keys(table: Mapping[str, Any], allowed: tuple[str, ...], where: str, fail: _Fail) -> None:
    """Refuse keys that are not in `allowed`, so a typo is not silently ignored."""
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        place = f" in [{where}]" if where else ""
        raise fail(f"unknown key(s){place}: {', '.join(unknown)}. The keys are: {', '.join(allowed)}.")


def _read_auth(table: Any, fail: _Fail) -> Optional[Auth]:
    """The [auth] table, or None if there is none."""
    if table is None:
        return None
    if not isinstance(table, dict):
        raise fail("'auth' must be a table, written as [auth] followed by its keys.")
    _refuse_unknown_keys(table, _AUTH_KEYS, "auth", fail)
    header = table.get("header")
    if not isinstance(header, str) or not _HEADER_PATTERN.fullmatch(header):
        raise fail("[auth] 'header' must be the name of an HTTP header, e.g. \"Authorization\".")
    env = table.get("env")
    if not isinstance(env, str) or not _ENV_PATTERN.fullmatch(env):
        raise fail("[auth] 'env' must be the name of an environment variable, e.g. \"ACME_TOKEN\".")
    prefix = table.get("prefix", "")
    if not isinstance(prefix, str):
        raise fail("[auth] 'prefix' must be text, e.g. \"Bearer \".")
    return Auth(header=header, env=env, prefix=prefix)


def _read_base_url(data: Mapping[str, Any], auth: Optional[Auth], fail: _Fail) -> str:
    """The base URL: http or https, a host, and no secret or query in it."""
    if "base_url" not in data:
        raise fail("the key 'base_url' is missing.")
    base_url = data["base_url"]
    if not isinstance(base_url, str):
        raise fail("'base_url' must be text, e.g. \"https://agent.example.com\".")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise fail("'base_url' must start with http:// or https:// and name a host.")
    if parts.username is not None or parts.password is not None:
        raise fail("'base_url' must not hold a user name or password. Put the secret in [auth].")
    if parts.query or parts.fragment:
        raise fail("'base_url' must not have a query or a fragment. A secret belongs in [auth].")
    if auth is not None and parts.scheme == "http" and not _is_loopback(parts.hostname):
        raise fail(
            "the auth header would go over plain http to another machine. "
            "Use https://, or http:// to localhost only."
        )
    return base_url


def _is_loopback(host: str) -> bool:
    """True for a host that is this machine: `localhost`, 127.x.x.x or ::1.

    `host` is a `urlsplit` hostname, which is lower case already.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _read_headers(table: Any, auth: Optional[Auth], fail: _Fail) -> dict[str, str]:
    """The [headers] table: names and text values, with no secret header among them."""
    if not isinstance(table, dict):
        raise fail("'headers' must be a table, written as [headers] followed by name = \"value\" lines.")
    for name, value in table.items():
        if not _HEADER_PATTERN.fullmatch(name):
            raise fail(f"[headers] {name!r} is not the name of an HTTP header.")
        if not isinstance(value, str):
            raise fail(f"[headers] {name!r} must be text.")
        if name.lower() in _SECRET_HEADERS:
            raise fail(f"[headers] {name!r} carries a secret, which must not be in the file. Use [auth].")
        if auth is not None and name.lower() == auth.header.lower():
            raise fail(f"[headers] {name!r} is also the [auth] header. Give it once.")
    return dict(table)


def _read_body(table: Any, fail: _Fail) -> dict[str, Any]:
    """The request body of the [request] table, or the default body if there is none."""
    if table is None:
        return dict(DEFAULT_BODY_TEMPLATE)
    if not isinstance(table, dict):
        raise fail("'request' must be a table, written as [request] followed by its keys.")
    _refuse_unknown_keys(table, ("body",), "request", fail)
    if "body" not in table:
        raise fail("[request] the key 'body' is missing.")
    body = table["body"]
    if not isinstance(body, dict):
        raise fail("[request] 'body' must be a table, e.g. body = { message = \"{message}\" }.")
    if _check_body_value(body, "request.body", fail) == 0:
        raise fail(f"[request] 'body' must have the value \"{MESSAGE_MARKER}\" somewhere, "
                   "or the customer's message is never sent.")
    return body


def _check_body_value(value: Any, where: str, fail: _Fail) -> int:
    """Check one value of the body and what is in it.

    Returns:
        How many times `{message}` is in it, as a whole value.

    Raises:
        TargetConfigError: For a marker that is part of a longer text, or a value
            that JSON cannot hold, such as a TOML date.
    """
    if isinstance(value, dict):
        return sum(_check_body_value(item, f"{where}.{key}", fail) for key, item in value.items())
    if isinstance(value, list):
        return sum(_check_body_value(item, f"{where}.{index}", fail) for index, item in enumerate(value))
    if isinstance(value, str):
        if value in MARKERS:
            return value == MESSAGE_MARKER
        for marker in MARKERS:
            if marker in value:
                raise fail(f"{where}: {value!r} has {marker} inside a longer text. "
                           f"A marker must be the whole value, e.g. \"{marker}\".")
        return 0
    if isinstance(value, (bool, int, float)):
        return 0
    raise fail(f"{where}: a {type(value).__name__} cannot be sent as JSON. Use text, a number or true/false.")


def _read_reply_path(table: Any, fail: _Fail) -> tuple[str, ...]:
    """The steps to the reply, from the [response] table, or the default if there is none."""
    if table is None:
        return DEFAULT_REPLY_PATH
    if not isinstance(table, dict):
        raise fail("'response' must be a table, written as [response] followed by its keys.")
    _refuse_unknown_keys(table, ("reply",), "response", fail)
    reply = table.get("reply")
    if not isinstance(reply, str) or not all(reply.split(".")):
        raise fail("[response] 'reply' must be the place of the reply, e.g. \"data.messages.0.text\".")
    return tuple(reply.split("."))


def _is_number(value: Any) -> bool:
    """True for a whole or decimal number, not for a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
