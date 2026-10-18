# mystery-ai-shopper
Intended for testing how your ai agent is good in using the tools you set up

## Layout

Everything specific to the system under test (here a fake shop and its support agent) lives in one package, so the testing code can be pointed at another system later.

| Folder | What it holds |
| --- | --- |
| `core/` | Reusable plumbing with no domain knowledge: configuration, shared types, and the Gemini agent loop (`agent_loop.py`). |
| `harness/` | Tools for testing an agent from outside, for now the tool-call recorder (`tool_recorder.py`). |
| `shop/` | The fake shop and everything that belongs to it: `store.py` and `seed.py` (the shop), `support_agent.py` (the agent's prompt and tools), `session.py` (one conversation with the agent), `app.py` (the HTTP API), `static/` (the demo page) and `tests/` (their tests). |

`core` imports nothing else from this project, and `harness` never imports `shop`.

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

## Tests

```
uv run pytest
```

The tests use a scripted fake in place of Gemini, so they need no API key and make no network calls.

Tests sit next to what they test: `tests/` holds the tests of the testing tool (`core`, `harness`) and the fakes they share (`tests/fakes.py`), and `shop/tests/` holds the tests of the fake shop and its support agent. `uv run pytest` runs both; add a folder name to run one, e.g. `uv run pytest shop/tests`.
