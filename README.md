# Gemma ADK Agent

Google ADK agent backed by Gemma 3 served locally via **Unsloth Studio**.

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
```

1. Open Unsloth Studio and load your preferred Gemma model.
2. Note the API base URL shown in the Studio UI (default `http://localhost:5000/v1`).
3. Set `UNSLOTH_BASE_URL` and `GEMMA_MODEL` in `.env`.

```bash
python main.py
```

## Recommended models (Unsloth HuggingFace IDs)

| Model | Quant | VRAM |
|-------|-------|------|
| `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` | 4-bit | ~3 GB |
| `unsloth/gemma-3-4b-it` | full | ~9 GB |
| `unsloth/gemma-3-12b-it-unsloth-bnb-4bit` | 4-bit | ~7 GB |
| `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` | 4-bit | ~15 GB |

The agent calls Unsloth Studio's OpenAI-compatible endpoint via LiteLLM, so no API key is needed.
