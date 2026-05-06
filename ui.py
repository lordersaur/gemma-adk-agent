import json
import os

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.text import Text
from rich.theme import Theme
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
})

console = Console(theme=_THEME, highlight=False)

PROMPT_STYLE = Style.from_dict({"prompt": "ansibrightcyan bold"})


def print_header(model_name: str) -> None:
    short = os.path.basename(model_name).replace(".gguf", "")
    console.print()
    console.print(Rule(f"[header]  Gemma ADK Agent  [/header][model]({short})[/model]", style="border"))
    console.print()


def print_user(text: str) -> None:
    console.print(Text("  You", style="user"), end="  ")
    console.print(text)
    console.print()


def make_thinking_status() -> Status:
    return Status("  thinking...", spinner="dots", console=console, spinner_style="dim")


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


def make_response_live() -> Live:
    return Live(
        "",
        console=console,
        refresh_per_second=15,
        vertical_overflow="visible",
    )


def render_response(text: str) -> Markdown:
    return Markdown(text)


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
    console.print(Markdown(text), style="agent")
    console.print()


def print_error(message: str) -> None:
    console.print(Text(f"  x  {message}", style="bold red"))
    console.print()


def print_warning(message: str) -> None:
    console.print(Text(f"  !  {message}", style="bold red"))


def print_success(message: str) -> None:
    console.print(Text(f"  ok  {message}", style="bold green"))
