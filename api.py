from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs" / "remote"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gemma Agent Log Server")


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
