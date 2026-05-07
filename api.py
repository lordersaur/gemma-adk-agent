from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import json

_LOG_DIR = Path(__file__).parent / "logs" / "remote"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gemma Agent Log Server")


class EventPayload(BaseModel):
    session_id: str
    record: dict


class TranscriptPayload(BaseModel):
    session_id: str
    text: str


@app.post("/log/event")
def log_event(payload: EventPayload):
    path = _LOG_DIR / f"{payload.session_id}_debug.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(payload.record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.post("/log/transcript")
def log_transcript(payload: TranscriptPayload):
    path = _LOG_DIR / f"{payload.session_id}_chat.md"
    with path.open("a") as f:
        f.write(payload.text)
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
