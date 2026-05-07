import asyncio
import json
import os
import shutil

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window, FloatContainer, Float
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style

from io import StringIO
from rich import box as richbox
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_THEME = Theme({
    "user":        "bold cyan",
    "agent":       "bold white",
    "thinking":    "dim italic",
    "tool.name":   "bold yellow",
    "tool.args":   "dim yellow",
    "tool.result": "dim green",
    "border":      "bright_black",
    "header":      "bold magenta",
    "model":       "dim magenta",
    "cmd":         "bold cyan",
    "cmd.desc":    "dim white",
})

PT_STYLE = Style.from_dict({
    "prompt":    "ansibrightcyan bold",
    "separator": "#444444",
})

_SLASH_COMMANDS = [
    ("/help",        "show help"),
    ("/clear",       "clear the screen"),
    ("/new",         "start a fresh session"),
    ("/history",     "show log file paths"),
    ("/model",       "show active model and backend"),
    ("/permissions", "show or clear auto-approved categories"),
    ("/exit",        "quit"),
]


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display=cmd, display_meta=desc)


class _Output:
    """Accumulates rendered ANSI chunks with support for replaceable slots."""

    def __init__(self):
        self._chunks: list[str] = []
        self._app: Application | None = None

    def attach(self, app: Application) -> None:
        self._app = app

    def _render(self, renderable) -> str:
        if isinstance(renderable, str):
            return renderable
        width = shutil.get_terminal_size((120, 24)).columns
        sio = StringIO()
        c = Console(file=sio, theme=_THEME, highlight=False, force_terminal=True, width=width)
        c.print(renderable)
        return sio.getvalue()

    def append(self, renderable) -> None:
        self._chunks.append(self._render(renderable))
        self._invalidate()

    def alloc_slot(self) -> int:
        idx = len(self._chunks)
        self._chunks.append("")
        return idx

    def update_slot(self, idx: int, renderable) -> None:
        if 0 <= idx < len(self._chunks):
            self._chunks[idx] = self._render(renderable)
            self._invalidate()

    def clear_slot(self, idx: int) -> None:
        if 0 <= idx < len(self._chunks):
            self._chunks[idx] = ""
            self._invalidate()

    def clear(self) -> None:
        self._chunks.clear()
        self._invalidate()

    def get_text(self):
        return ANSI("".join(self._chunks))

    def get_cursor(self):
        text = "".join(self._chunks)
        return Point(x=0, y=text.count("\n"))

    def _invalidate(self) -> None:
        if self._app:
            self._app.invalidate()


# ── Module-level state ────────────────────────────────────────────────────────

_output = _Output()
_pt_app: Application | None = None
_input_queue: asyncio.Queue[str] = asyncio.Queue()
_interrupt_event: asyncio.Event = asyncio.Event()
_input_buf: Buffer | None = None

# Slot indices for live updates
_thinking_slot: int | None = None
_stream_slot: int | None = None
_terminal_slot: int | None = None


# ── App construction ──────────────────────────────────────────────────────────

def _build_app() -> Application:
    global _input_buf

    _input_buf = Buffer(
        completer=SlashCompleter(),
        complete_while_typing=True,
        name="input",
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event):
        text = _input_buf.text.strip()
        _input_buf.reset()
        if text:
            _input_queue.put_nowait(text)

    @kb.add("c-c")
    def _ctrl_c(event):
        _interrupt_event.set()

    @kb.add("c-d")
    def _ctrl_d(event):
        event.app.exit()

    @kb.add("up")
    def _up(event):
        _input_buf.history_backward()

    @kb.add("down")
    def _down(event):
        _input_buf.history_forward()

    output_win = Window(
        content=FormattedTextControl(
            text=_output.get_text,
            focusable=False,
            show_cursor=False,
            get_cursor_position=_output.get_cursor,
        ),
        wrap_lines=True,
    )

    sep = Window(height=1, char="─", style="class:separator")

    input_win = Window(
        content=BufferControl(buffer=_input_buf, focusable=True),
        height=1,
        get_line_prefix=lambda lineno, wrap_count: [("class:prompt", " > ")],
    )

    layout = Layout(
        FloatContainer(
            content=HSplit([output_win, sep, input_win]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                ),
            ],
        )
    )
    app = Application(layout=layout, key_bindings=kb, style=PT_STYLE, full_screen=True)
    _output.attach(app)
    return app


async def run_app() -> None:
    global _pt_app
    _pt_app = _build_app()
    await _pt_app.run_async()


async def get_input() -> str:
    return await _input_queue.get()


def exit_app() -> None:
    if _pt_app:
        _pt_app.exit()


# ── Print helpers ─────────────────────────────────────────────────────────────

def write_markup(text: str) -> None:
    _output.append(Text.from_markup(text))


def print_header(model_name: str) -> None:
    short = os.path.basename(model_name).replace(".gguf", "")
    _output.append(Rule(f"[header]  Gemma ADK Agent  [/header][model]({short})[/model]", style="border"))
    _output.append("")


def print_user(text: str) -> None:
    t = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    t.add_column(style="white on #0a1628")
    t.add_row(text)
    _output.append(t)
    _output.append("")


def print_thinking(text: str) -> None:
    if not text.strip():
        return
    n_lines = len(text.strip().splitlines())
    _output.append(Panel(
        Text(text.strip(), style="thinking"),
        title=f"[dim]thinking · {n_lines} lines[/dim]",
        border_style="bright_black",
        padding=(0, 1),
    ))
    _output.append("")


def print_thinking_summary(elapsed_seconds: int) -> None:
    _output.append(Text(f"  thought for {elapsed_seconds}s", style="dim italic"))
    _output.append("")


def print_tool_call(name: str, args: dict) -> None:
    try:
        arg_str = json.dumps(args, ensure_ascii=False)
        arg_str = arg_str[:120] + ("..." if len(arg_str) > 120 else "")
    except Exception:
        arg_str = str(args)
    _output.append(
        Text("  > ", style="tool.name") +
        Text(name, style="tool.name") +
        Text(f"  {arg_str}", style="tool.args")
    )


def print_tool_result(name: str, result: str) -> None:
    preview = result.strip().splitlines()[0][:80] if result.strip() else "(done)"
    _output.append(Text(f"    -> {preview}", style="tool.result"))
    _output.append("")


def print_response(text: str) -> None:
    if not text:
        return
    _output.append(Markdown("● " + text))
    _output.append("")


def print_error(message: str) -> None:
    _output.append(Text(f"  x  {message}", style="bold red"))
    _output.append("")


def print_warning(message: str) -> None:
    _output.append(Text(f"  !  {message}", style="bold yellow"))


def print_success(message: str) -> None:
    _output.append(Text(f"  ok  {message}", style="bold green"))


def print_help() -> None:
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column(style="cmd", no_wrap=True)
    table.add_column(style="cmd.desc")
    for cmd, desc in [
        ("/help",        "show this help"),
        ("/clear",       "clear the screen"),
        ("/new",         "start a fresh session (resets model context)"),
        ("/history",     "show current log file paths"),
        ("/model",       "show active model and backend"),
        ("/permissions", "show or clear auto-approved command categories"),
        ("/exit",        "quit"),
    ]:
        table.add_row(cmd, desc)
    _output.append(Panel(table, title="[header]Commands[/header]", border_style="bright_black", padding=(0, 1)))
    _output.append("")


def print_history(jsonl_path, transcript_path) -> None:
    _output.append(Text(f"  jsonl      {jsonl_path}", style="dim"))
    _output.append(Text(f"  transcript {transcript_path}", style="dim"))
    _output.append("")


def print_model_info(model: str, base_url: str) -> None:
    _output.append(Text(f"  model    {model}", style="dim"))
    _output.append(Text(f"  backend  {base_url}", style="dim"))
    _output.append("")


def print_permissions(categories: set[str]) -> None:
    if not categories:
        _output.append(Text("  no auto-approved categories", style="dim"))
    else:
        for cat in sorted(categories):
            _output.append(Text(f"  + {cat}", style="bold green"))
    _output.append("")


def print_session_reset(old_tokens: int, new_session_id: str) -> None:
    _output.append(Rule(
        f"[dim]  context reset  ({old_tokens:,} tokens)  ->  {new_session_id}  [/dim]",
        style="yellow"
    ))
    _output.append("")


def clear_screen() -> None:
    _output.clear()


# ── Live update slots ─────────────────────────────────────────────────────────

def start_thinking_status() -> None:
    global _thinking_slot
    _thinking_slot = _output.alloc_slot()


def update_thinking_status(elapsed: int) -> None:
    if _thinking_slot is not None:
        _output.update_slot(_thinking_slot, Text(f"  thinking... {elapsed}s", style="dim italic"))


def stop_thinking_status() -> None:
    global _thinking_slot
    if _thinking_slot is not None:
        _output.clear_slot(_thinking_slot)
        _thinking_slot = None


def start_stream() -> None:
    global _stream_slot
    _stream_slot = _output.alloc_slot()


def update_stream(text: str) -> None:
    if _stream_slot is not None:
        _output.update_slot(_stream_slot, Markdown("● " + text))


def end_stream() -> None:
    global _stream_slot
    _stream_slot = None


def start_terminal_progress() -> None:
    global _terminal_slot
    _terminal_slot = _output.alloc_slot()


def update_terminal_progress(lines: list[str], done: bool = False) -> None:
    if _terminal_slot is None:
        return
    status = f"({len(lines)} lines)" if done else "running..."
    preview = lines[-2:] if done else lines[-20:]
    body = ""
    for l in preview:
        if l.strip():
            body += f"  {l.rstrip()}\n"
    _output.update_slot(_terminal_slot, f"  terminal  {status}\n{body}")


def end_terminal_progress() -> None:
    global _terminal_slot
    _terminal_slot = None


# ── Confirmation ──────────────────────────────────────────────────────────────

async def confirm_terminal(command: str, category: str = "destructive") -> str:
    is_destructive = category == "destructive"
    style = "bold red" if is_destructive else "bold yellow"
    prefix = "  ! " if is_destructive else "  ? "
    _output.append(Text(f"{prefix}{command}", style=style))
    if is_destructive:
        _output.append(Text("    [y] run  [n] cancel  ", style="dim"))
    else:
        _output.append(Text(f"    [y] run  [n] cancel  [r] remember all {category}  ", style="dim"))
    answer = await get_input()
    _output.append(Text(f"  > {answer}", style="dim"))
    return answer.strip().lower()
