# mystery-ai-shopper
Intended for testing how your ai agent is good in using the tools you set up

## Setup

The agent reads the Anthropic (Claude) API key from the `ANTHROPIC_API_KEY` system environment variable. The key is never stored in the repository.

Windows (PowerShell), permanent for your user:

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Linux/macOS, permanent for your user (add to `~/.bashrc`, or `~/.zshrc` for zsh, which is the macOS default):

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.zshrc
```

Restart the terminal/IDE afterwards so it picks up the variable.

## Run

```
uv run fastapi dev main.py
```
