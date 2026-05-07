import os
import platform
import requests
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from tools import web_search, python, terminal

load_dotenv()

UNSLOTH_BASE_URL = os.getenv("UNSLOTH_BASE_URL", "http://localhost:8080/v1")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-E4B-it-UD-Q8_K_XL")
API_KEY = os.getenv("UNSLOTH_API_KEY", "")

_base_url = UNSLOTH_BASE_URL.replace("/v1", "")


def _auth_headers() -> dict:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def ensure_model_loaded() -> None:
    try:
        resp = requests.get(f"{_base_url}/health", headers=_auth_headers(), timeout=5)
        if resp.ok:
            print(f"[motor] server ready at {UNSLOTH_BASE_URL}")
            return
    except requests.RequestException:
        pass
    raise RuntimeError(f"Backend not reachable at {_base_url} — is the server running?")


def _build_instruction() -> str:
    cwd = os.getcwd()
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    env_info = f"{system} {release} ({machine})"
    return (
        "<|think|>"
        "thinking: LOW\n"
        f"Local dev assistant. OS: {env_info}. CWD: {cwd}.\n"
        "Tools:\n"
        "- web_search: docs, APIs, packages.\n"
        "- python: run code, write files. "
        f"Write files with: content = '''...''' then open(abs_path, 'w').write(content). "
        f"Always absolute paths (e.g. {cwd}/proj/src/App.jsx). "
        "Never f-strings for JSX/TSX content (curly braces break JSON encoding) — plain triple-quoted strings only. "
        "Never print() instead of writing — open().write() or it didn't happen. "
        "On write failure rewrite the whole file, never assume partial success.\n"
        "- terminal: shell commands, scaffolding, installs, git. "
        "CWD persists across calls — cd once and it stays. "
        "Pass --yes/-y/--force to skip interactive prompts. "
        "Remove existing dirs before scaffolding into them. "
        "Never ls -R (node_modules overflow) — use ls -F or find . -maxdepth 2 -not -path '*/node_modules/*'. "
        "Never run dev servers (npm start/run dev block forever) — give the user the command and URL instead "
        f"(Vite: http://localhost:5173, CRA: http://localhost:3000).\n"
        "Behavior: act, don't advise. Use tools immediately — never announce then wait. "
        "Keep going until the task is done. Only stop when blocked or intent is ambiguous."
    )


# ── Model + Agent ──────────────────────────────────────────────────────────────

_model = LiteLlm(
    model=f"openai/{GEMMA_MODEL}",
    api_base=UNSLOTH_BASE_URL,
    api_key=API_KEY or "unsloth",
)

root_agent = LlmAgent(
    name="gemma_agent",
    model=_model,
    description="A local development assistant powered by Gemma 4.",
    instruction=_build_instruction(),
    tools=[web_search, python, terminal],
)
