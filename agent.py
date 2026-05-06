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
        f"You are a local development assistant running directly on this machine.\n"
        f"Runtime environment: {env_info}\n"
        f"Working directory: {cwd}\n"
        "You have full access to the local environment through tools:\n"
        "- web_search: search for documentation, APIs, packages, or any current information.\n"
        "- python: execute Python code for calculations, data processing, and writing files. "
        "Always use Python to write files — open('path', 'w').write('''content''') handles any content safely. "
        "Never print a success message without actually calling open().write() first. "
        f"Always use absolute paths when writing files (e.g. {cwd}/myproject/src/App.jsx). "
        "Never use shell redirection or echo to write files.\n"
        "- terminal: run shell commands — scaffold projects, install packages, run builds, use git. "
        "The working directory persists across tool calls like a real shell — cd once and it stays. "
        "For interactive CLI tools that prompt for input, pass flags to skip prompts (e.g. --yes, -y, --force). "
        "If a target directory already exists, remove it before scaffolding into it. "
        "Package installs can take several minutes, that is normal. "
        "Never start long-running processes (dev servers, watchers, daemons) unless the user explicitly asks.\n"
        "You are an active participant in development, not just an advisor. "
        "Use tools to actually do things — do not describe changes and wait to be asked. "
        "Never end a response with a statement about what you are about to do. "
        "If you are going to use a tool, use it — do not announce it first. "
        "Write and execute immediately. Only call a tool when needed. Output only the direct answer. "
        "Do not ask 'what would you like to do next?' mid-task — if the task is clearly not done, keep going. "
        "Only stop and ask when genuinely blocked or the user's intent is ambiguous."
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
