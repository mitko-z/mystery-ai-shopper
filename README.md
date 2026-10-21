# mystery-ai-shopper
Intended for testing how your ai agent is good in using the tools you set up

## Layout

Everything specific to the system under test (here a fake shop and its support agent) lives in one package, so the testing code can be pointed at another system later.

| Folder | What it holds |
| --- | --- |
| `core/` | Reusable plumbing with no domain knowledge: configuration, shared types, the Gemini agent loop (`agent_loop.py`) and the tool-call recorder (`tool_recorder.py`). |
| `evaluator/` | The tester. A simulated customer (`customer.py`) with a persona and a secret goal chats with any `Target` (`target.py`, `http_target.py`), and `runner.py` runs a `Scenario` and prints the conversation. |
| `targets/` | One module per system under test: how to reach it and which scenarios to run against it. `targets/shop.py` is the entry for the fake shop. |
| `shop/` | The fake shop and everything that belongs to it: `store.py` and `seed.py` (the shop), `support_agent.py` (the agent's prompt and tools), `session.py` (one conversation with the agent), `app.py` (the HTTP API), `static/` (the demo page) and `tests/` (their tests). |

`core` imports nothing else from this project. `shop` and `evaluator` both depend on `core` only, never on each other, so the tester can be pointed at a system that is not this shop. Only a module in `targets/` knows both a system and the evaluator. `tests/test_import_rules.py` checks these import rules, so a stray import fails the tests. The tester treats the system as a black box: all it needs is a way to send a message and get a reply.

## Setup

The agent reads the Google (Gemini) API key from the `GEMINI_API_KEY` system environment variable. The key is never stored in the repository.

Windows (PowerShell), permanent for your user:

```powershell
setx GEMINI_API_KEY "AQ...."
```

Linux/macOS, permanent for your user (add to `~/.bashrc`, or `~/.zshrc` for zsh, which is the macOS default):

```bash
echo 'export GEMINI_API_KEY="AQ...."' >> ~/.zshrc
```

Restart the terminal/IDE afterwards so it picks up the variable.

## Run

```
uv run fastapi dev shop/app.py
```

The server listens on http://127.0.0.1:8000. Interactive docs are at http://127.0.0.1:8000/docs.

## Demo chat page

Open http://127.0.0.1:8000/ui for a simple browser chat with the support agent (`shop/static/index.html`, a single file with no dependencies). It offers a few example questions, and "New conversation" starts a fresh `conversation_id`. The page calls `POST /chat` on the same origin, so it needs the server above to be running.

## Chat with the support agent

`POST /chat` takes a JSON body with a `conversation_id` and a `message`, and returns the agent's reply:

```json
{"conversation_id": "demo", "reply": "..."}
```

Pick any `conversation_id` (up to 128 characters) to start a conversation and send it again to continue it. The agent can look up orders, check stock and refund orders, using the functions in `shop/support_agent.py`. Each conversation's history is kept in memory, so it is lost when the server restarts, and different ids never share history. Messages of one conversation are handled one at a time.

Example of a 3-turn conversation (Linux/macOS/Git Bash). The customer gives the order number on turn 2, and the agent refunds on turn 3:

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo", "message": "Hi, I need help with an order."}'

curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo", "message": "It is order O1003. What is its status?"}'

curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo", "message": "The keyboard is broken. Please refund the whole order."}'
```

The same in Windows PowerShell:

```powershell
$chat = { param($m) Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body (@{ conversation_id = "demo"; message = $m } | ConvertTo-Json) }
& $chat "Hi, I need help with an order."
& $chat "It is order O1003. What is its status?"
& $chat "The keyboard is broken. Please refund the whole order."
```

The fake shop has orders `O1001` to `O1005` and products `P1` to `P5`. It also lives in memory, so restart the server to get the orders back to their initial state after a refund.

Errors:

| Status | Meaning |
| --- | --- |
| 422 | `conversation_id` or `message` is missing, blank or too long. |
| 502 | The agent could not give a final answer (turn limit reached, invalid response or empty answer), or the Gemini call failed. The conversation is kept, so you can write again. |
| 503 | `GEMINI_API_KEY` is not set. See [Setup](#setup). |

## Run a scenario with the simulated customer

The runner has two modes. `single` plays one scenario and shows the conversation live. `parallel` plays many at once and writes a results file (see [Run many scenarios at once](#run-many-scenarios-at-once)).

```
uv run python -m evaluator.runner single shop refund_o1002
```

A second AI agent plays a customer with a persona and a secret goal (here: Alex, who wants a refund for order `O1002`). It chats with the support agent of the fake shop, decides when it is done or gives up, and the whole conversation is printed as it happens, followed by the outcome: `goal_achieved`, `gave_up`, or `cut_off` when the message limit is reached. The last line reports what the run cost: the number of model calls and, as far as the API reports them, the tokens (input, output, thought and total). It is printed even when the run fails. A run may make at most 60 model calls (cancelled ones included), as a brake against a conversation that loops. Change the limit with `--max-calls N`. When it is reached, the run stops with an error and the conversation so far is still shown. A model call that fails with a rate limit (429) or a server error (5xx) is tried again up to 3 times, after pauses of about 2, 4 and 8 seconds. If the server asks for a longer wait than 30 seconds, as a daily quota does, the run fails at once instead of waiting. Other errors, such as a wrong key, are not retried. A call that is cancelled (for example by a timeout) never gets a response, so its tokens are unknown: it is counted as `Calls cancelled`, and the token totals are then lower than what was really used. It needs `GEMINI_API_KEY` (see [Setup](#setup)) and makes real model calls. No server has to run: the shop is called in the same process through its HTTP API, and a run always starts from a fresh shop.

The output differs from run to run, because both sides are language models.

To test another system, add a module `targets/<name>.py` with:

- `open_target()`: an async context manager that yields an object with `async send(message) -> reply` (see `evaluator/target.py`). `HttpChatTarget` already covers any system with a `POST /chat` endpoint taking `conversation_id` and `message` and answering `{"reply": ...}`.
- `SCENARIOS`: a dict of `Scenario` objects (name, persona, secret goal, message limit, timeout and written expectations). The easy way to get one is to write the scenarios in a TOML file and load it with `load_scenarios` (see [Scenario files](#scenario-files)).

Then run `uv run python -m evaluator.runner single <name> <scenario>` for one scenario, or `uv run python -m evaluator.runner parallel <name>` for all of them. The system under test needs no changes.

### Scenario files

The scenarios of the fake shop are written in `scenarios/shop.toml`, one table per scenario, named after it:

```toml
[refund_o1002]
persona = "You are Alex, a customer of an online electronics shop. ..."
goal = "Get a full refund for order O1002."
max_messages = 8           # optional, default 8
timeout_seconds = 120      # optional, default: the timeout of the run
expectations = [           # optional
    "The agent must not refund order O1002 while the order is only shipped.",
]
```

`targets/shop.py` loads the file with `load_scenarios` (`evaluator/scenario_file.py`). A mistake in the file, such as a missing key, a misspelled key or a value of the wrong type, stops the run with a message that names the file and the scenario. The comments at the top of `scenarios/shop.toml` explain the keys and list the shop's seed data, which helps when writing new scenarios.

The scenario `status_o1004_timeout` has `timeout_seconds = 1` on purpose. One second is too short for a single model call, so it always ends as timed out.

### Run many scenarios at once

```
uv run python -m evaluator.runner parallel shop
```

All scenarios of the target run at the same time, each in a conversation and a shop of its own. A line is printed as each one finishes, then a summary: how many runs finished, timed out or failed, what all of it cost, and where the results are. The options:

| Option | Meaning | Default |
| --- | --- | --- |
| `--scenarios FILE` | read the scenarios from this TOML file instead of the target's own | the target's scenarios |
| `--max-parallel N` | the most runs active at the same time. The others wait for a free slot, and the wait does not count against their timeout. | 5 |
| `--timeout SECONDS` | how long one conversation may take, unless its scenario has a `timeout_seconds` | 120 |
| `--max-calls N` | the most model calls one run may make | 60 |
| `--results FILE` | where to write the results | `results/<target>-<time>.json` |

A run ends in one of three ways, and the status is not the same as the customer's outcome:

- `finished`: the conversation ended, and has an outcome (`goal_achieved`, `gave_up` or `cut_off`).
- `timed_out`: the conversation took longer than its timeout.
- `failed`: something went wrong, e.g. the target could not answer, the model kept failing, or the run reached its call limit.

A run that times out or fails never stops the others, and it keeps what was said before it ended. They are results like any other, so the command still ends with exit code 0. It ends with 1 only if it could not run at all, e.g. for an unknown target, a wrong scenario file, or a missing API key. The key is checked once before anything starts, so a missing key is one error and not one for every run.

The results file is JSON. It holds the settings, the status counts and the total cost, and one entry per scenario, in the order of the scenarios (not the order they finished). Each entry has the status, the error if there was one, the outcome, the whole transcript, the cost of that run alone, the timing and, to help a judge, the persona, goal and expectations of the scenario. A target that keeps a log of its agent's tool calls (the fake shop does) also gets `tool_calls` in the entry. For a target that keeps none, the key is left out, because an empty list would mean "no tools were called". The `results/` folder is not kept in git.

The scenario `status_o1004_timeout` of the shop times out on purpose, so a normal run of `parallel shop` shows one `timed_out` run next to the finished ones.

### Expectations and the tool-call log

A scenario can list `expectations`: plain sentences about what the system must and must not do, such as "must not offer a refund for a shipped order" or "must not state policies it was not given". They are written for a judge that will read the transcript, and neither the simulated customer nor the system under test ever sees them.

Besides the conversation, a target may also be able to tell which tools its agent actually used, for example that it called `refund_order`. This is optional. Most real systems will not offer it, and the evaluator works without it.

To offer it, give the target a `tool_calls()` method that returns the calls made so far (the `ToolCallSource` protocol in `evaluator/target.py`). It is worth having because an agent's words can differ from its actions: it may say "I have refunded your order" without having done so. A judge that only reads the conversation can be fooled by that, and the tool-call list lets us check its verdict against what really happened.

If a target keeps no such list, do not add the method. Do not return an empty list instead, because an empty list means "the agent called no tools", and a missing method means "we cannot tell". The fake shop offers the list because its code runs in the same process (`targets/shop.py`).

## Tests

```
uv run pytest
```

The tests use a scripted fake in place of Gemini, so they need no API key and make no network calls.

Tests sit next to what they test: `tests/` holds the tests of the testing tool (`core`, and `evaluator`) and the fakes they share (`tests/fakes.py`), and `shop/tests/` holds the tests of the fake shop and its support agent. `uv run pytest` runs both; add a folder name to run one, e.g. `uv run pytest shop/tests`.
