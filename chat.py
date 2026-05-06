import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from prompt_toolkit import PromptSession

from agent import ensure_model_loaded
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
    ui.print_header(model_name)

    prompt_session: PromptSession = PromptSession()

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: prompt_session.prompt(
                    [("class:prompt", " ❯ ")],
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

        ui.print_user(user_input)
        await _send(runner, user_input)


async def _send(runner: Runner, user_input: str) -> None:
    message = Content(role="user", parts=[Part(text=user_input)])

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=message
        ):
            for fc in event.get_function_calls():
                ui.print_tool_call(fc.name, dict(fc.args) if fc.args else {})

            for fr in event.get_function_responses():
                result = ""
                if fr.response:
                    result = str(fr.response.get("output", fr.response))
                ui.print_tool_result(fr.name, result)

            if event.is_final_response() and event.content:
                text = "".join(p.text for p in event.content.parts if p.text).strip()
                if text:
                    ui.print_response(text)

    except Exception as e:
        err = str(e)
        if "502" in err or "Lost connection" in err or "crashed" in err:
            ui.print_warning("model server crashed — reloading…")
            try:
                ensure_model_loaded()
                ui.print_success("model reloaded, please resend your message")
            except Exception as reload_err:
                ui.print_error(f"reload failed: {reload_err}")
        else:
            ui.print_error(err)
