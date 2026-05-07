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
        "Act, don't advise — use tools immediately, never announce then wait. "
        "Never show code in a response and call it done — write the file, confirm in one line. "
        "After writing any Python file, always run python3 -m py_compile <file> via terminal to verify syntax. "
        "When web_search returns an official or docs URL relevant to the question, always fetch it before answering — snippets are never complete. "
        "If the fetched page does not contain the answer, fetch a more specific URL or search again — never answer from incomplete content. "
        "Never use emojis. "
        "Keep going until done. Stop only if blocked or intent is ambiguous."
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
