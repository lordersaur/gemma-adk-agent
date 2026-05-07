"""
Combined server: handles /log/* locally, proxies everything else to llama.cpp.
Run on port 8079:  uvicorn api:app --host 0.0.0.0 --port 8079
Cloudflare tunnel: ./cloudflared tunnel --url http://localhost:8079

Remote .env:
  UNSLOTH_BASE_URL=https://<tunnel>.trycloudflare.com/v1
  LOG_SERVER_URL=https://<tunnel>.trycloudflare.com
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080")

_LOG_DIR = Path(__file__).parent / "logs" / "remote"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()


# ── Log endpoints ─────────────────────────────────────────────────────────────

class SessionPayload(BaseModel):
    session_id: str
    jsonl: str
    transcript: str


@app.post("/log/session")
def upload_session(payload: SessionPayload):
    (_LOG_DIR / f"{payload.session_id}_debug.jsonl").write_text(payload.jsonl)
    (_LOG_DIR / f"{payload.session_id}_chat.md").write_text(payload.transcript)
    return {"ok": True}


@app.get("/log/sessions")
def list_sessions():
    sessions = sorted({
        p.name.replace("_debug.jsonl", "")
        for p in _LOG_DIR.glob("*_debug.jsonl")
    })
    return {"sessions": sessions}


@app.get("/log/session/{session_id}/transcript")
def get_transcript(session_id: str):
    path = _LOG_DIR / f"{session_id}_chat.md"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return JSONResponse(content={"transcript": path.read_text()})


# ── Proxy to llama.cpp ────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(
        method=request.method,
        url=f"{LLAMA_URL}/{path}",
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        content=await request.body(),
        params=dict(request.query_params),
    )
    resp = await client.send(req, stream=True)

    async def stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
