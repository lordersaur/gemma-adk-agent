import asyncio
import time

from rich.live import Live
from rich.status import Status

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from agent import ensure_model_loaded
import logger
import ui

APP_NAME = "gemma-adk-agent"
USER_ID = "local_user"
SESSION_ID = "session_001"


async def run(model_name: str, app_name: str = APP_NAME) -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=USER_ID, session_id=SESSION_ID
    )

    from agent import root_agent
    runner = Runner(agent=root_agent, app_name=app_name, session_service=session_service)

    ensure_model_loaded()
    session_id = logger.init_session()
    jsonl, transcript = logger.session_paths()

    ui.print_header(model_name)
    ui.console.print(f"[dim]  logs -> {transcript.name}[/dim]\n")

    prompt_session: PromptSession = PromptSession()

    with patch_stdout():
        await _input_loop(runner, prompt_session)


async def _input_loop(runner: Runner, prompt_session: PromptSession) -> None:
    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: prompt_session.prompt(
                    [("class:prompt", " > ")],
                    style=ui.PROMPT_STYLE,
                ).strip(),
            )
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]bye[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            ui.console.print("[dim]bye[/dim]")
            break

        logger.log_user(user_input)
        task = asyncio.ensure_future(_send(runner, user_input))
        try:
            await task
        except (KeyboardInterrupt, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            ui.console.print("\n[dim]interrupted[/dim]\n")


def _unwrap_tool_result(response: dict) -> str:
    if not response:
        return ""
    val = response.get("output") or response.get("result") or response
    if isinstance(val, dict):
        val = val.get("output") or val.get("result") or str(val)
    return str(val)


async def _send(runner: Runner, user_input: str) -> None:
    message = Content(role="user", parts=[Part(text=user_input)])
    thinking_buf = ""
    response_buf = ""
    usage_metadata = None
    t_start = time.monotonic()

    _status: Status | None = None
    _live: Live | None = None
    _thinking_displayed = False
    _response_streamed = False

    def _stop_status() -> None:
        nonlocal _status
        if _status is not None:
            _status.stop()
            _status = None

    def _stop_live() -> None:
        nonlocal _live
        if _live is not None:
            _live.stop()
            _live = None

    def _flush_thinking() -> None:
        nonlocal _thinking_displayed
        if thinking_buf and not _thinking_displayed:
            logger.log_thinking(thinking_buf)
            ui.print_thinking(thinking_buf)
            _thinking_displayed = True

    # Start waiting indicator
    _status = ui.make_thinking_status()
    _status.start()

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=message
        ):
            for fc in event.get_function_calls():
                _stop_status()
                _stop_live()
                args = dict(fc.args) if fc.args else {}
                logger.log_tool_call(fc.name, args)
                ui.print_tool_call(fc.name, args)

            for fr in event.get_function_responses():
                result = _unwrap_tool_result(fr.response)
                logger.log_tool_result(fr.name, result)
                ui.print_tool_result(fr.name, result)
                # Restart thinking indicator for next model turn
                if _status is None and _live is None:
                    _status = ui.make_thinking_status()
                    _status.start()

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
                    thinking_buf += think_text

                if resp_text and event.partial:
                    # First response token — end thinking phase, start streaming
                    if _status is not None or not _thinking_displayed:
                        _stop_status()
                        _flush_thinking()
                    if _live is None:
                        _live = ui.make_response_live()
                        _live.start()
                        _response_streamed = True
                    response_buf += resp_text
                    _live.update(ui.render_response(response_buf))

            if event.is_final_response():
                _stop_status()

                # Fallback: if no partial events, collect from final event
                if not _thinking_displayed and not thinking_buf and event.content:
                    t = "".join(
                        p.text for p in event.content.parts
                        if p.text and getattr(p, "thought", False)
                    )
                    if t:
                        thinking_buf = t

                _flush_thinking()

                if not _response_streamed and event.content:
                    r = "".join(
                        p.text for p in event.content.parts
                        if p.text and not getattr(p, "thought", False)
                    )
                    if r:
                        response_buf = r

                _stop_live()

    except Exception as e:
        _stop_status()
        _stop_live()
        err = str(e)
        logger.log_error(err)
        if "502" in err or "Lost connection" in err or "crashed" in err:
            ui.print_warning("model server crashed — reloading...")
            try:
                ensure_model_loaded()
                ui.print_success("model reloaded, please resend your message")
            except Exception as reload_err:
                ui.print_error(f"reload failed: {reload_err}")
        else:
            ui.print_error(err)
        return

    elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

    if usage_metadata:
        pt = getattr(usage_metadata, "prompt_token_count", 0) or 0
        ct = getattr(usage_metadata, "candidates_token_count", 0) or 0
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
        # Print if we didn't stream it
        if not _response_streamed:
            ui.print_response(response_buf)
    else:
        ui.console.print()
