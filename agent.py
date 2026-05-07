import os
import platform
import requests
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from tools import web_search, python, terminal, write_file, edit_file

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
        "Act, don't advise — use tools immediately, never announce then wait. "
        "Never show code in a response and call it done — write the file, confirm in one line. "
        "Always use write_file to create or overwrite files, and edit_file to modify part of an existing file — never use echo, printf, or heredoc in terminal for file writing. "
        "Use the python tool only for calculations and running code, not for writing files. "
        "After writing any .py file, always run python3 -m py_compile <file> via terminal to verify syntax — never run py_compile on .jsx, .js, .ts, or any non-Python file. "
        "Never start a dev server or long-running process (npm run dev, vite, flask run, uvicorn, node, etc.) — write the files, then tell the user the exact command to run. "
        "If the search snippets do not fully answer the question, fetch the most relevant URL from the results before answering — do not guess or answer from incomplete content. "
        "For CLI tools available locally, run <command> --help via terminal instead of searching the web — it is faster and more accurate. "
        "Never use emojis. "
        "If a tool returns 'Command cancelled by user', stop immediately and tell the user the command was not run — never claim the operation succeeded. "
        "Keep going until done. Stop only if blocked or intent is ambiguous."
    )


# ── Model + Agent ──────────────────────────────────────────────────────────────

_model = LiteLlm(
    model=f"openai/{GEMMA_MODEL}",
    api_base=UNSLOTH_BASE_URL,
    api_key=API_KEY or "unsloth",
    temperature=0.2,
)

root_agent = LlmAgent(
    name="gemma_agent",
    model=_model,
    description="A local development assistant powered by Gemma 4.",
    instruction=_build_instruction(),
    tools=[web_search, python, terminal, write_file, edit_file],
)
