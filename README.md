# Gemma ADK Agent

A local AI coding assistant built on Google ADK + Gemma 4, served via **llama.cpp**. Features a full-screen terminal UI with a pinned input bar, streaming responses, tool use, and persistent session logging.

![terminal UI with split layout — output top, input pinned at bottom]

## Features

- Full-screen TUI with output scrollback and input always at the bottom
- Streaming responses with live thinking timer
- Tools: web search + URL fetch, Python execution, shell terminal (with CWD persistence), file write, file edit
- Tiered permission system for shell commands (destructive / install / execute) with persistent allow-list
- Automatic context reset with session summary injection when approaching the context limit
- Session logs: JSONL + human-readable transcript per session

## Requirements

- Python 3.12+
- A running llama.cpp server (or any OpenAI-compatible backend)
- A Gemma 4 GGUF model

## Setup

```bash
git clone https://github.com/lordersaur/gemma-adk-agent
cd gemma-adk-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
UNSLOTH_BASE_URL=http://localhost:8080/v1
GEMMA_MODEL=gemma-4-E4B-it-UD-Q8_K_XL
```

## Running the server (llama.cpp)

The agent expects an OpenAI-compatible HTTP server. The companion server config lives at `../llama-cpp-motor/`. A minimal `.env` for the server:

```env
MODEL_PATH=/path/to/your/gemma-4.gguf
HOST=0.0.0.0
PORT=8080
GPU_LAYERS=-1
CTX_SIZE=65536
REASONING_BUDGET=2048
PARALLEL=1
```

Recommended model: `gemma-4-E4B-it-UD-Q8_K_XL.gguf` (~9 GB VRAM on a single GPU).

## Running the agent

```bash
python main.py
```

Or install as a CLI entry point (if configured in your shell):

```bash
gemma-cli
```

## Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show command list |
| `/clear` | Clear the screen |
| `/new` | Start a fresh session (resets model context) |
| `/history` | Show current log file paths |
| `/model` | Show active model and backend URL |
| `/permissions` | Show or clear auto-approved command categories |
| `/exit` | Quit |

Keyboard shortcuts: **Ctrl+C** interrupts a running response, **Ctrl+D** quits.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search or direct URL fetch |
| `python` | Execute Python code (calculations, data processing) |
| `terminal` | Run shell commands with CWD persistence across calls |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace an exact string in an existing file |

Shell commands that match destructive, install, or execute patterns require confirmation before running. Approvals can be remembered per category.

## Connecting from another machine

The server binds to `0.0.0.0` by default. To use the agent from another PC:

**Local network:**
```bash
# find this machine's LAN IP
ip addr show | grep 'inet ' | grep -v 127.0.0.1
# open firewall port if needed
sudo ufw allow 8080/tcp
```
Then set `UNSLOTH_BASE_URL=http://<lan-ip>:8080/v1` on the remote machine.

**Over the internet (no port forwarding needed):**
```bash
cloudflared tunnel --url http://localhost:8080
```
Use the printed `https://*.trycloudflare.com` URL as `UNSLOTH_BASE_URL` (append `/v1`).

If the server is exposed beyond your LAN, set `UNSLOTH_API_KEY` in both the server and agent `.env` files.

## Project structure

```
agent.py      — LlmAgent definition, model config, system prompt
chat.py       — main session loop, streaming, context reset logic
ui.py         — prompt_toolkit full-screen TUI
tools.py      — tool implementations and permission system
logger.py     — JSONL + transcript session logging
api.py        — FastAPI wrapper (optional HTTP interface)
main.py       — entry point
```
