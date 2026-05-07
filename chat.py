import asyncio
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from agent import ensure_model_loaded
import logger
import ui

APP_NAME = "gemma-adk-agent"
USER_ID = "local_user"
CONTEXT_RESET_THRESHOLD = 55_000


async def _new_session(session_service: InMemorySessionService, app_name: str) -> str:
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=USER_ID, session_id=session_id
    )
    return session_id


async def run(model_name: str, app_name: str = APP_NAME) -> None:
    session_service = InMemorySessionService()
    session_id = await _new_session(session_service, app_name)

    from agent import root_agent
    runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)

    ensure_model_loaded()
    from agent import UNSLOTH_BASE_URL
    logger.init_session(model=model_name, base_url=UNSLOTH_BASE_URL)
    _, transcript = logger.session_paths()

    ui.print_header(model_name)
    ui.write_markup(f"[dim]  logs -> {transcript.name}[/dim]\n")

    loop_task = asyncio.ensure_future(
        _input_loop(runner, session_service, session_id, app_name, model_name)
    )
    await ui.run_app()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


async def _input_loop(
    runner: Runner,
    session_service: InMemorySessionService,
    session_id: str,
    app_name: str,
    model_name: str,
) -> None:
    last_prompt_tokens = 0

    while True:
        ui._interrupt_event.clear()
        user_input = await ui.get_input()

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/exit", "/quit"):
            logger.flush_session()
            ui.write_markup("[dim]bye[/dim]")
            ui.exit_app()
            return
        elif cmd == "/help":
            ui.print_help()
            continue
        elif cmd == "/clear":
            ui.clear_screen()
            ui.print_header(model_name)
            continue
        elif cmd == "/new":
            session_id = await _new_session(session_service, app_name)
            logger.log_context_reset(session_id)
            last_prompt_tokens = 0
            ui.clear_screen()
            ui.print_header(model_name)
            ui.print_success("new session — context cleared")
            continue
        elif cmd == "/history":
            jsonl, transcript = logger.session_paths()
            ui.print_history(jsonl, transcript)
            continue
        elif cmd == "/model":
            from agent import UNSLOTH_BASE_URL
            ui.print_model_info(model_name, UNSLOTH_BASE_URL)
            continue
        elif cmd == "/permissions":
            from tools import _auto_approved_categories, _save_permissions
            ui.print_permissions(_auto_approved_categories)
            if _auto_approved_categories:
                ui.write_markup("[dim]  clear all? [y/N][/dim]")
                answer = await ui.get_input()
                if answer.strip().lower() == "y":
                    _auto_approved_categories.clear()
                    _save_permissions(_auto_approved_categories)
                    ui.print_success("permissions cleared")
            continue

        logger.log_user(user_input)
        ui.print_user(user_input)

        send_task = asyncio.ensure_future(_send(runner, session_id, user_input))

        async def _watch_interrupt():
            await ui._interrupt_event.wait()
            send_task.cancel()

        watcher = asyncio.ensure_future(_watch_interrupt())
        prompt_tokens = 0
        try:
            prompt_tokens = await send_task
        except asyncio.CancelledError:
            ui.write_markup("[dim]interrupted[/dim]\n")
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        if prompt_tokens:
            last_prompt_tokens = prompt_tokens

        if last_prompt_tokens >= CONTEXT_RESET_THRESHOLD:
            summary = logger.build_summary(max_exchanges=20)
            old_tokens = last_prompt_tokens
            session_id = await _new_session(session_service, app_name)
            logger.log_context_reset(session_id)
            last_prompt_tokens = 0
            ui.print_session_reset(old_tokens, session_id)
            if summary:
                await _send(runner, session_id, f"[Context reset — previous session summary]\n{summary}")


def _unwrap_tool_result(response: dict) -> str:
    if not response:
        return ""
    val = response.get("output") or response.get("result") or response
    if isinstance(val, dict):
        val = val.get("output") or val.get("result") or str(val)
    return str(val)


_PARSE_ERROR_MARKERS = (
    "Unterminated string",
    "JSONDecodeError",
    "json.decoder",
    "Expecting value",
    "Invalid control character",
    "Invalid \\escape",
)


def _is_tool_call_parse_error(err: str) -> bool:
    return any(m.lower() in err.lower() for m in _PARSE_ERROR_MARKERS)


async def _send(runner: Runner, session_id: str, user_input: str) -> int:
    message = Content(role="user", parts=[Part(text=user_input)])
    thinking_buf = ""
    response_buf = ""
    usage_metadata = None
    t_start = time.monotonic()
    thinking_start: float | None = None

    _thinking_shown = False
    _response_streamed = False

    _timer_task: list[asyncio.Task | None] = [None]

    ui.start_thinking_status()

    async def _run_timer() -> None:
        await asyncio.sleep(2.0)
        try:
            while True:
                elapsed = round(time.monotonic() - (thinking_start or t_start))
                ui.update_thinking_status(elapsed)
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            pass

    def _cancel_timer() -> None:
        if _timer_task[0] is not None and not _timer_task[0].done():
            _timer_task[0].cancel()
            _timer_task[0] = None
        ui.stop_thinking_status()

    def _flush_thinking() -> None:
        nonlocal _thinking_shown
        if _thinking_shown:
            return
        _thinking_shown = True
        elapsed = round(time.monotonic() - (thinking_start or t_start))
        if thinking_buf:
            logger.log_thinking(thinking_buf)
        if elapsed >= 2:
            ui.print_thinking_summary(elapsed)

    _timer_task[0] = asyncio.ensure_future(_run_timer())

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=message
        ):
            for fc in event.get_function_calls():
                _cancel_timer()
                ui.end_stream()
                _flush_thinking()
                args = dict(fc.args) if fc.args else {}
                logger.log_tool_call(fc.name, args)
                ui.print_tool_call(fc.name, args)

            for fr in event.get_function_responses():
                result = _unwrap_tool_result(fr.response)
                logger.log_tool_result(fr.name, result)
                ui.print_tool_result(fr.name, result)
                if _timer_task[0] is None:
                    ui.start_thinking_status()
                    _timer_task[0] = asyncio.ensure_future(_run_timer())

            if event.usage_metadata:
                usage_metadata = event.usage_metadata

            if event.content and event.content.parts:
                think_text = "".join(
                    p.text for p in event.content.parts
                    if p.text and getattr(p, "thought", False)
                )
                resp_text = "".join(
                    p.text for p in event.content.parts
                    if p.text and not getattr(p, "thought", False)
                )

                if think_text:
                    if thinking_start is None:
                        thinking_start = time.monotonic()
                    thinking_buf += think_text

                if resp_text and event.partial:
                    _cancel_timer()
                    _flush_thinking()
                    if not _response_streamed:
                        ui.start_stream()
                        _response_streamed = True
                    response_buf += resp_text
                    ui.update_stream(response_buf)

            if event.is_final_response():
                _cancel_timer()

                if not _thinking_shown and not thinking_buf and event.content:
                    t = "".join(
                        p.text for p in event.content.parts
                        if p.text and getattr(p, "thought", False)
                    )
                    if t:
                        thinking_buf = t
                        thinking_start = thinking_start or t_start

                _flush_thinking()

                if not _response_streamed and event.content:
                    r = "".join(
                        p.text for p in event.content.parts
                        if p.text and not getattr(p, "thought", False)
                    )
                    if r:
                        response_buf = r

                ui.end_stream()

    except Exception as e:
        _cancel_timer()
        ui.end_stream()
        err = str(e)
        logger.log_error(err)
        if "502" in err or "Lost connection" in err or "crashed" in err:
            ui.print_warning("model server crashed — reloading...")
            try:
                ensure_model_loaded()
                ui.print_success("model reloaded, please resend your message")
            except Exception as reload_err:
                ui.print_error(f"reload failed: {reload_err}")
        elif _is_tool_call_parse_error(err):
            ui.print_warning(f"tool call JSON parse error: {err}")
            recovery = (
                f"Your last tool call could not be executed because the arguments contained "
                f"characters that broke JSON encoding (error: {err}). "
                "This usually happens when Python code contains raw newlines, unescaped quotes, "
                "or curly braces inside an f-string. "
                "Fix: assign the file content to a variable using plain triple-quoted string "
                "(content = '''...'''), then call open(path, 'w').write(content). "
                "Do NOT use f-strings for file content. Retry the operation now."
            )
            logger.log_error(f"[auto-recovery] injecting: {recovery}")
            return await _send(runner, session_id, recovery)
        else:
            ui.print_error(err)
        return 0

    elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
    prompt_tokens = 0

    if usage_metadata:
        pt = getattr(usage_metadata, "prompt_token_count", 0) or 0
        ct = getattr(usage_metadata, "candidates_token_count", 0) or 0
        prompt_tokens = pt
        stats = {
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "tokens_per_sec": round(ct / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0,
        }
        logger.log_llm_stats(stats)

    if response_buf:
        logger.log_raw_chunks([response_buf])
        logger.log_response(response_buf)
        if not _response_streamed:
            ui.print_response(response_buf)
        else:
            ui.write_markup("")
    else:
        ui.write_markup("")

    return prompt_tokens
