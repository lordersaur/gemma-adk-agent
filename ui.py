import json
import os

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
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


def print_thinking(text: str) -> None:
    if not text:
        return
    console.print(Text("  thinking", style="thinking"))
    for line in text.splitlines():
        console.print(Text(f"    {line}", style="thinking"))
    console.print()


def print_tool_call(name: str, args: dict) -> None:
    try:
        arg_str = json.dumps(args, ensure_ascii=False)
        arg_str = arg_str[:120] + ("…" if len(arg_str) > 120 else "")
    except Exception:
        arg_str = str(args)
    console.print(
        Text("  ⚙ ", style="tool.name") +
        Text(name, style="tool.name") +
        Text(f"  {arg_str}", style="tool.args")
    )


def print_tool_result(name: str, result: str) -> None:
    preview = result.strip().splitlines()[0][:80] if result.strip() else "(done)"
    console.print(Text(f"    ↳ {preview}", style="tool.result"))
    console.print()


def print_response(text: str) -> None:
    if not text:
        return
    console.print(Markdown(text), style="agent")
    console.print()


def print_error(message: str) -> None:
    console.print(Text(f"  ✗  {message}", style="bold red"))
    console.print()


def print_warning(message: str) -> None:
    console.print(Text(f"  ⚠  {message}", style="bold red"))


def print_success(message: str) -> None:
    console.print(Text(f"  ✓  {message}", style="bold green"))
