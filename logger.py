import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_session_id: str = ""
_jsonl_path: Path | None = None
_transcript_path: Path | None = None
_log_server_url: str | None = None


def _upload(sid: str, jsonl: Path, transcript: Path) -> None:
    try:
        import requests
        requests.post(
            f"{_log_server_url}/log/session",
            json={
                "session_id": sid,
                "jsonl": jsonl.read_text() if jsonl.exists() else "",
                "transcript": transcript.read_text() if transcript.exists() else "",
            },
            timeout=10,
        )
    except Exception:
        pass


def _upload_previous() -> None:
    if not _log_server_url or not _session_id or not _jsonl_path:
        return
    sid, j, t = _session_id, _jsonl_path, _transcript_path
    threading.Thread(target=_upload, args=(sid, j, t), daemon=True).start()


def init_session(model: str = "", base_url: str = "") -> str:
    global _session_id, _jsonl_path, _transcript_path, _log_server_url
    _log_server_url = os.getenv("LOG_SERVER_URL", "").rstrip("/") or None
    _upload_previous()
    _session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_DIR.mkdir(exist_ok=True)
    _jsonl_path = _LOG_DIR / f"{_session_id}_debug.jsonl"
    _transcript_path = _LOG_DIR / f"{_session_id}_chat.md"
    model = model or os.getenv("GEMMA_MODEL", "unknown")
    base_url = base_url or os.getenv("UNSLOTH_BASE_URL", "")
    _write_transcript(f"# Session {_session_id}\n\n**Model:** `{model}`  \n**Backend:** `{base_url}`\n\n---\n\n")
    _event("session_start", {"model": model, "base_url": base_url})
    return _session_id


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, data: dict) -> None:
    if _jsonl_path is None:
        return
    record = {"ts": _ts(), "type": type_, **data}
    with _jsonl_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_transcript(text: str) -> None:
    if _transcript_path is None:
        return
    with _transcript_path.open("a") as f:
        f.write(text)


# ── public logging API ─────────────────────────────────────────────────────────

def log_user(text: str) -> None:
    _event("user", {"text": text})
    _write_transcript(f"### You\n\n{text}\n\n")


def log_tool_call(name: str, args: dict) -> None:
    _event("tool_call", {"name": name, "args": args})
    args_str = json.dumps(args, ensure_ascii=False, indent=2)
    _write_transcript(f"#### Tool call: `{name}`\n\n```json\n{args_str}\n```\n\n")


def log_tool_result(name: str, result: str) -> None:
    _event("tool_result", {"name": name, "result": result})
    _write_transcript(f"#### Tool result: `{name}`\n\n```\n{result[:2000]}\n```\n\n")


def log_raw_chunks(chunks: list[str]) -> None:
    """Log every final-response chunk exactly as received from the model."""
    _event("llm_raw_chunks", {"chunks": chunks, "count": len(chunks)})
    for i, chunk in enumerate(chunks, 1):
        _write_transcript(f"<details><summary>raw chunk {i}/{len(chunks)}</summary>\n\n```\n{chunk}\n```\n\n</details>\n\n")


def log_llm_stats(stats: dict) -> None:
    """Log token counts and timing captured from the LiteLLM callback."""
    _event("llm_stats", stats)
    rows = "\n".join(f"| {k} | {v} |" for k, v in stats.items())
    _write_transcript(f"#### Stats\n\n| key | value |\n|-----|-------|\n{rows}\n\n")


def log_thinking(text: str) -> None:
    _event("thinking", {"text": text})
    _write_transcript(f"<details><summary>thinking</summary>\n\n{text}\n\n</details>\n\n")


def log_response(text: str) -> None:
    _event("response", {"text": text})
    _write_transcript(f"### Agent\n\n{text}\n\n---\n\n")


def log_error(message: str) -> None:
    _event("error", {"message": message})
    _write_transcript(f"> **Error:** {message}\n\n")


def log_context_reset(new_session_id: str) -> None:
    _event("context_reset", {"new_session_id": new_session_id})
    _write_transcript(f"> **Context reset** → session `{new_session_id}`\n\n")


def build_summary(max_exchanges: int = 20) -> str:
    """Read the current JSONL log and produce a compact summary of recent exchanges."""
    if _jsonl_path is None or not _jsonl_path.exists():
        return ""
    exchanges: list[dict] = []
    current: dict = {}
    with _jsonl_path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = record.get("type")
            if t == "user":
                if current:
                    exchanges.append(current)
                current = {"user": record.get("text", ""), "response": "", "tools": []}
            elif t == "tool_call" and current:
                current["tools"].append(record.get("name", ""))
            elif t == "response" and current:
                current["response"] = record.get("text", "")
    if current:
        exchanges.append(current)

    recent = exchanges[-max_exchanges:]
    lines = []
    for ex in recent:
        tools_note = f" [tools: {', '.join(ex['tools'])}]" if ex["tools"] else ""
        lines.append(f"User: {ex['user']}{tools_note}")
        if ex["response"]:
            snippet = ex["response"][:200].replace("\n", " ")
            lines.append(f"Agent: {snippet}")
    return "\n".join(lines)


def flush_session() -> None:
    """Upload the current session to the log server. Call on exit."""
    _upload_previous()


def session_paths() -> tuple[Path, Path]:
    """Return (jsonl_path, transcript_path) for the current session."""
    return _jsonl_path, _transcript_path
