import json
import os
import sys

from rich import box as richbox
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style

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

console = Console(theme=_THEME, highlight=False)

PROMPT_STYLE = Style.from_dict({
    "prompt":  "ansibrightcyan bold bg:ansimagenta",
    "completion-menu.completion":              "bg:#2a1a3e white",
    "completion-menu.completion.current":      "bg:ansimagenta white bold",
    "completion-menu.meta.completion":         "bg:#2a1a3e #888888",
    "completion-menu.meta.completion.current": "bg:ansimagenta #cccccc",
})

_SLASH_COMMANDS = [
    ("/help",    "show help"),
    ("/clear",   "clear the screen"),
    ("/new",     "start a fresh session"),
    ("/history", "show log file paths"),
    ("/model",   "show active model and backend"),
    ("/exit",    "quit"),
]


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


def print_header(model_name: str) -> None:
    short = os.path.basename(model_name).replace(".gguf", "")
    console.print()
    console.print(Rule(f"[header]  Gemma ADK Agent  [/header][model]({short})[/model]", style="border"))
    console.print()


def print_user(text: str) -> None:
    console.print(Panel(Text(text), box=richbox.MINIMAL, style="white on #0a1628", padding=(0, 0)))
    console.print()


def make_thinking_live() -> Live:
    return Live("", console=console, refresh_per_second=8, transient=True)


def render_thinking_status(elapsed: int) -> Text:
    return Text(f"  thinking... {elapsed}s", style="dim italic")


def print_thinking(text: str) -> None:
    if not text.strip():
        return
    n_lines = len(text.strip().splitlines())
    panel = Panel(
        Text(text.strip(), style="thinking"),
        title=f"[dim]thinking · {n_lines} lines[/dim]",
        border_style="bright_black",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def print_thinking_summary(elapsed_seconds: int) -> None:
    console.print(Text(f"  thought for {elapsed_seconds}s", style="dim italic"))
    console.print()


def make_response_live() -> Live:
    return Live(
        "",
        console=console,
        refresh_per_second=15,
        vertical_overflow="visible",
    )


def render_response(text: str) -> Markdown:
    return Markdown("● " + text)


def render_terminal_live(lines: list[str], done: bool = False) -> object:
    from rich.console import Group
    status = f"({len(lines)} lines)" if done else "running..."
    header = Rule(f"[dim]terminal  {status}[/dim]", style="bright_black")
    preview = lines[-2:] if done else lines[-20:]
    body = [Text(f"  {l.rstrip()}", style="dim") for l in preview if l.strip()]
    return Group(header, *body)


def print_tool_call(name: str, args: dict) -> None:
    try:
        arg_str = json.dumps(args, ensure_ascii=False)
        arg_str = arg_str[:120] + ("..." if len(arg_str) > 120 else "")
    except Exception:
        arg_str = str(args)
    console.print(
        Text("  > ", style="tool.name") +
        Text(name, style="tool.name") +
        Text(f"  {arg_str}", style="tool.args")
    )


def print_tool_result(name: str, result: str) -> None:
    preview = result.strip().splitlines()[0][:80] if result.strip() else "(done)"
    console.print(Text(f"    -> {preview}", style="tool.result"))
    console.print()


def print_response(text: str) -> None:
    if not text:
        return
    console.print(render_response(text))
    console.print()


async def confirm_terminal(command: str, base: str, is_destructive: bool = False) -> str:
    import asyncio
    style = "bold red" if is_destructive else "bold yellow"
    prefix = "  ! " if is_destructive else "  ? "
    console.print(Text(f"{prefix}{command}", style=style))
    console.print(Text(f"    [y] run  [n] cancel  [r] remember '{base}'  ", style="dim"), end="")
    loop = asyncio.get_event_loop()
    return (await loop.run_in_executor(None, sys.stdin.readline)).strip().lower()


def print_error(message: str) -> None:
    console.print(Text(f"  x  {message}", style="bold red"))
    console.print()


def print_warning(message: str) -> None:
    console.print(Text(f"  !  {message}", style="bold yellow"))


def print_success(message: str) -> None:
    console.print(Text(f"  ok  {message}", style="bold green"))


def print_help() -> None:
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column(style="cmd", no_wrap=True)
    table.add_column(style="cmd.desc")
    commands = [
        ("/help",    "show this help"),
        ("/clear",   "clear the screen"),
        ("/new",     "start a fresh session (resets model context)"),
        ("/history", "show current log file paths"),
        ("/model",   "show active model and backend"),
        ("/exit",    "quit"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    panel = Panel(
        table,
        title="[header]Commands[/header]",
        border_style="bright_black",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def print_history(jsonl_path, transcript_path) -> None:
    console.print(Text(f"  jsonl      {jsonl_path}", style="dim"))
    console.print(Text(f"  transcript {transcript_path}", style="dim"))
    console.print()


def print_model_info(model: str, base_url: str) -> None:
    console.print(Text(f"  model    {model}", style="dim"))
    console.print(Text(f"  backend  {base_url}", style="dim"))
    console.print()


def print_session_reset(old_tokens: int, new_session_id: str) -> None:
    console.print(Rule(
        f"[dim]  context reset  ({old_tokens:,} tokens)  ->  {new_session_id}  [/dim]",
        style="yellow"
    ))
    console.print()
