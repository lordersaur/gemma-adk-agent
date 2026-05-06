import os
import requests
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from tools import web_search, python, terminal

load_dotenv()

UNSLOTH_BASE_URL = os.getenv("UNSLOTH_BASE_URL", "http://localhost:8888/v1")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-E4B-it-UD-Q8_K_XL")
GGUF_PATH = os.getenv("GGUF_PATH", "")
API_KEY = os.getenv("UNSLOTH_API_KEY", "")

_studio_base = UNSLOTH_BASE_URL.replace("/v1", "")


def _auth_headers() -> dict:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def ensure_model_loaded() -> None:
    """Load the GGUF into Unsloth Studio if it isn't already loaded."""
    try:
        status = requests.get(f"{_studio_base}/api/inference/status", headers=_auth_headers(), timeout=5)
        if status.ok:
            data = status.json()
            loaded = data.get("loaded", [])
            active = data.get("active_model", "")
            if GGUF_PATH in loaded or GGUF_PATH == active:
                print(f"[unsloth] model already loaded: {active}")
                return
    except requests.RequestException:
        pass

    if not GGUF_PATH:
        print("[unsloth] GGUF_PATH not set — assuming model is already loaded in Studio")
        return

    print(f"[unsloth] loading {GGUF_PATH} ...")
    resp = requests.post(
        f"{_studio_base}/api/inference/load",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={"model_path": GGUF_PATH},
        timeout=300,
    )
    if not resp.ok:
        raise RuntimeError(f"Failed to load model: {resp.status_code} {resp.text}")
    print("[unsloth] model loaded.")


_model = LiteLlm(
    model=f"openai/{GEMMA_MODEL}",
    api_base=UNSLOTH_BASE_URL,
    api_key=API_KEY or "unsloth",
)

root_agent = LlmAgent(
    name="gemma_agent",
    model=_model,
    description="A helpful assistant powered by Gemma 4 E4B running locally via Unsloth Studio.",
    instruction=(
        "You are a helpful, concise assistant with access to tools.\n"
        "- Use web_search to look up current information.\n"
        "- Use python to execute Python code and return results.\n"
        "- Use terminal to run shell commands.\n"
        "Only call a tool when it is actually needed."
    ),
    tools=[web_search, python, terminal],
)
