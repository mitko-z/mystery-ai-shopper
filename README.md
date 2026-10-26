# mystery-ai-shopper
Intended for testing how your ai agent is good in using the tools you set up

## Layout

Everything specific to the system under test (here a fake shop and its support agent) lives in one package, so the testing code can be pointed at another system later.

| Folder | What it holds |
| --- | --- |
| `core/` | Reusable plumbing with no domain knowledge: configuration, shared types, the Gemini agent loop (`agent_loop.py`) and the tool-call recorder (`tool_recorder.py`). |
| `evaluator/` | The tester. A simulated customer (`customer.py`) with a persona and a secret goal chats with any `Target` (`target.py`, `http_target.py`), and `runner.py` runs a `Scenario` and records the conversation. `judge.py` judges whether the target met the expectations of a scenario, `agreement.py` compares the judge with our own verdicts, and `cli.py` is the command line for all of it (`python -m evaluator`). |
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

The command line has four commands, started as `python -m evaluator <command>`. `single` plays one scenario and shows the conversation live. `parallel` plays many at once and writes a results file (see [Run many scenarios at once](#run-many-scenarios-at-once)). `judge` and `agreement` work on such a file afterwards (see [Judge a results file](#judge-a-results-file)).

```
uv run python -m evaluator single shop refund_o1002
```

A second AI agent plays a customer with a persona and a secret goal (here: Alex, who wants a refund for order `O1002`). It chats with the support agent of the fake shop, decides when it is done or gives up, and the whole conversation is printed as it happens, followed by the outcome: `goal_achieved`, `gave_up`, or `cut_off` when the message limit is reached. The last line reports what the run cost: the number of model calls and, as far as the API reports them, the tokens (input, output, thought and total). It is printed even when the run fails. A run may make at most 60 model calls (cancelled ones included), as a brake against a conversation that loops. Change the limit with `--max-calls N`. When it is reached, the run stops with an error and the conversation so far is still shown. A model call that fails with a rate limit (429) or a server error (5xx) is tried again up to 3 times, after pauses of about 2, 4 and 8 seconds. If the server asks for a longer wait than 30 seconds, as a daily quota does, the run fails at once instead of waiting. Other errors, such as a wrong key, are not retried. A call that is cancelled (for example by a timeout) never gets a response, so its tokens are unknown: it is counted as `Calls cancelled`, and the token totals are then lower than what was really used. It needs `GEMINI_API_KEY` (see [Setup](#setup)) and makes real model calls. No server has to run: the shop is called in the same process through its HTTP API, and a run always starts from a fresh shop.

The output differs from run to run, because both sides are language models.

To test another system, add a module `targets/<name>.py` with:

- `open_target()`: an async context manager that yields an object with `async send(message) -> reply` (see `evaluator/target.py`). `HttpChatTarget` already covers any system with a `POST /chat` endpoint taking `conversation_id` and `message` and answering `{"reply": ...}`.
- `SCENARIOS`: a dict of `Scenario` objects (name, persona, secret goal, message limit, timeout and written expectations). The easy way to get one is to write the scenarios in a TOML file and load it with `load_scenarios` (see [Scenario files](#scenario-files)).

Then run `uv run python -m evaluator single <name> <scenario>` for one scenario, or `uv run python -m evaluator parallel <name>` for all of them. The system under test needs no changes.

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

The scenario `status_o1004_timeout` has `timeout_seconds = 1` on purpose. One second is too short for a single model call, so it always ends as timed out. It is kept in a file of its own, `scenarios/shop_timeout.toml`, so that a normal batch does not hold a run that is sure to time out. Run it with `--scenario-file scenarios/shop_timeout.toml`.

### Run many scenarios at once

**Warning:** this makes many real model calls (about 6 to 8 per scenario) and uses your API quota.

```
uv run python -m evaluator parallel shop
```

All scenarios of the target run at the same time, each in a conversation and a shop of its own. A line is printed as each one finishes, then a summary: how many runs finished, timed out or failed, what all of it cost, and where the results are. The options:

| Option | Meaning | Default |
| --- | --- | --- |
| `--scenario-file FILE` | read the scenarios from this TOML file instead of the target's own. Repeat the option to use several files, e.g. `--scenario-file a.toml --scenario-file b.toml`. A scenario name must not be in two files. | the target's scenarios |
| `--max-parallel N` | the most runs active at the same time. The others wait for a free slot, and the wait does not count against their timeout. | 5 |
| `--timeout SECONDS` | how long one conversation may take, unless its scenario has a `timeout_seconds` | 120 |
| `--max-calls N` | the most model calls one run may make | 60 |
| `--results FILE` | where to write the results | `results/<target>-<time>.json` |

A run ends in one of three ways, and the status is not the same as the customer's outcome:

- `finished`: the conversation ended, and has an outcome (`goal_achieved`, `gave_up` or `cut_off`).
- `timed_out`: the conversation took longer than its timeout.
- `failed`: something went wrong, e.g. the target could not answer, the model kept failing, or the run reached its call limit.

A run that times out or fails never stops the others, and it keeps what was said before it ended. The results file is written all the same, because those runs are part of the result. The key is checked once before anything starts, so a missing key is one error and not one for every run. The command tells how it went by its exit code:

| Exit code | Meaning |
| --- | --- |
| 0 | every run finished. What the customer decided (`goal_achieved`, `gave_up`, `cut_off`) is for the judge to weigh, so it does not change the code. |
| 1 | it could not run at all, e.g. an unknown target, a wrong scenario file, a missing API key, or a results file that cannot be written |
| 2 | the command line is wrong (argparse) |
| 3 | it ran and wrote the file, but at least one run failed or timed out. This is also what a wrong API key looks like, because the key is only checked to be set: every run fails. |
| 130 | interrupted with Ctrl+C |

The results file is JSON. It holds the settings, the status counts and the total cost, and one entry per scenario, in the order of the scenarios (not the order they finished). Each entry has the status, the error if there was one, the outcome, the whole transcript, the cost of that run alone, the timing, the timeout the run had (a scenario can have one of its own, so it is not always the default in `settings`) and, to help a judge, the persona, goal and expectations of the scenario. A target that keeps a log of its agent's tool calls (the fake shop does) also gets `tool_calls` in the entry. For a target that keeps none, the key is left out, because an empty list would mean "no tools were called". The `results/` folder is not kept in git.

To see a `timed_out` run, play the scenario that times out on purpose together with the others: `uv run python -m evaluator parallel shop --scenario-file scenarios/shop.toml --scenario-file scenarios/shop_timeout.toml`. It ends with exit code 3.

### Expectations and the tool-call log

A scenario can list `expectations`: plain sentences about what the system must and must not do, such as "must not offer a refund for a shipped order" or "must not state policies it was not given". They are written for a judge that will read the transcript, and neither the simulated customer nor the system under test ever sees them.

Besides the conversation, a target may also be able to tell which tools its agent actually used, for example that it called `refund_order`. This is optional. Most real systems will not offer it, and the evaluator works without it.

To offer it, give the target a `tool_calls()` method that returns the calls made so far (the `ToolCallSource` protocol in `evaluator/target.py`). It is worth having because an agent's words can differ from its actions: it may say "I have refunded your order" without having done so. A judge that only reads the conversation can be fooled by that, and the tool-call list lets us check its verdict against what really happened.

If a target keeps no such list, do not add the method. Do not return an empty list instead, because an empty list means "the agent called no tools", and a missing method means "we cannot tell". The fake shop offers the list because its code runs in the same process (`targets/shop.py`).

## Judge a results file

**Warning:** this makes real model calls (about 2 for each expectation) and uses your API quota.

```
uv run python -m evaluator judge results/shop-20261023-185334.json
```

The judge reads a results file and, for every run that `finished`, decides for each expectation of its scenario whether the target met it. Each expectation is judged in a short conversation of its own with a model. The model sees that one expectation and the transcript, and reports `pass`, `fail` or `cannot_tell` through a tool, with the evidence it relies on and its reasoning. `cannot_tell` is the honest answer when the conversation does not show enough, for example whether the agent really looked something up first. The judge never sees the persona, the goal, the customer's own outcome, the other expectations or anything of our own verdicts. A customer message at the very end of a transcript is the goodbye that was never sent to the target, and is marked so.

With `--tool-log` the judge also gets the target's log of tool calls, for the runs that have one. Without it, or for a run without a log, it works from the conversation alone. Runs that did not finish, or whose scenario has no expectations, are not judged. They are listed in the file with the reason.

| Option | Meaning | Default |
| --- | --- | --- |
| `--tool-log` | also give the judge the tool log of each run that has one | off |
| `--out FILE` | where to write the judgements | `evaluations/<results name>.judge-v1.json`, or `.judge-v1-toollog.json` with `--tool-log` |
| `--max-parallel N` | the most judgements active at the same time | 5 |
| `--timeout SECONDS` | how long one judgement may take | 60 |
| `--max-calls N` | the most model calls one judgement may make | 6 |

A judge file is never overwritten, and this is checked before any model call. The first judgement of a version of the rules is kept: change the rules, and the new file gets a new version in its name. The file records the model, the version, the mode and the full text the judge was told, the cost, the runs that were left out, and a verdict with its evidence and reasoning for every expectation. The exit code is 0 when every expectation got a verdict, 1 when it could not run (a wrong file, nothing to judge, a file that exists, a missing API key), 2 for a wrong command line, 3 when at least one judgement got no verdict (the file is written all the same) and 130 after Ctrl+C. A verdict that was reported stands, also if an error came after it.

### Compare the judge with our own verdicts

```
uv run python -m evaluator agreement evaluations/shop-20261023-185334.ground-truth.json evaluations/shop-20261023-185334.judge-v1.json
```

The key, `evaluations/<results name>.ground-truth.json`, holds our own verdict (`pass`, `fail` or `unclear`) for each expectation of one results file, with a line of evidence. A person writes it, with the tool log and the agent's instruction at hand, before the judge is run. `needs_tool_log` marks an expectation whose verdict depends on something the judge cannot see.

The command writes a Markdown table with a row for each expectation: our verdict, the judge's, and whether they agree. A `cannot_tell` of the judge is expected where the expectation needs the tool log, and a miss anywhere else. The summary counts the rows with and without the duplicate row (`covered_by`), and separately for the expectations that the transcript is enough for and those that need the log. A row that needs a cause (a disagreement, a missing verdict or a missed `cannot_tell`) says `TODO` in its last cell, for a person to replace with a line. Because of that, an existing table is kept unless you add `--overwrite`. `--out FILE` names the table; by default it goes beside the judge file, with `.judge-` replaced by `.agreement-`. The command makes no model call. It exits with 0, or with 1 if the files cannot be read, do not belong together, or the table exists.

### Controls and repeated runs

A judge can be wrong in two ways that a table of real runs may not show. Its answers vary from run to run, so run each mode a few times and look at the rows that change. And if every policy expectation of the real runs is a `fail`, a judge that fails every sentence which sounds like a policy would look perfect. So `evaluations/<name>.controls.json` holds controls: hand-edited copies of real runs in which the agent behaves well (and one in which it adds a single invented sentence), with their own key, `<name>.controls.ground-truth.json`. They have the shape of a results file, so the same commands work on them:

```
uv run python -m evaluator judge evaluations/shop-20261023-185334.controls.json
uv run python -m evaluator agreement evaluations/shop-20261023-185334.controls.ground-truth.json evaluations/shop-20261023-185334.controls.judge-v1.json
```

A judge file is never overwritten, so a repeat of the same judging needs a name of its own: `--out evaluations/shop-20261023-185334.judge-v1-run2.json`. `evaluations/shop-20261023-185334.runs-v1.md` puts the three runs of each mode side by side for the real runs and for the controls. It was made by a one-off script, not by a command.

## Tests

```
uv run pytest
```

The tests use a scripted fake in place of Gemini, so they need no API key and make no network calls.

Tests sit next to what they test: `tests/` holds the tests of the testing tool (`core`, and `evaluator`) and the fakes they share (`tests/fakes.py`), and `shop/tests/` holds the tests of the fake shop and its support agent. `uv run pytest` runs both; add a folder name to run one, e.g. `uv run pytest shop/tests`.
