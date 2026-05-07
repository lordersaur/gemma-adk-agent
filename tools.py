import asyncio
import re
import subprocess
import sys
import tempfile
import os
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def _fetch_page(url: str, max_chars: int = 8000) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except Exception as e:
        return f"(could not read page: {e})"


def web_search(query: str, max_results: int = 3) -> str:
    """Search the web by query, or fetch a specific URL directly.

    TWO MODES:
    - Query mode: returns titles, URLs, and short snippets. Snippets are
      summaries only — always fetch the most relevant URL before answering
      questions about docs, flags, APIs, or anything requiring a complete list.
      A snippet mentioning one fact is not a complete answer. If a docs or
      official URL is in the results, fetch it.
    - URL mode: pass a full URL (https://...) to fetch and read the full page.

    Args:
        query: Search query string, or a full URL to fetch directly.
        max_results: Number of results for search mode (default 3).

    Returns:
        Page content for URL fetches, or titles + snippets for searches.
    """
    query = query.strip()
    if query.startswith(("http://", "https://")):
        content = _fetch_page(query)
        return f"[fetched] {query}\n\n{content}"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}\n{r['href']}\n{r.get('body', '')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def python(code: str) -> str:
    """Execute Python code and return stdout + stderr.

    IMPORTANT: this tool runs in its own temp directory, not the terminal CWD.
    Always use absolute paths when writing files: open('/abs/path/file.py', 'w').
    File content must be a plain triple-quoted string — no f-strings with curly
    braces (JSX/JSON break encoding), no docstrings inside the content string.
    No emojis in file content, print statements, comments, or strings.
    If the call errors, retry with corrected code — never claim success on failure.
    Never use this tool to run code instead of writing a file, and never use it
    to preview or draft code — write directly to the file on the first call.
    Always write to an absolute path first (open('/abs/path/file.py', 'w').write(content)),
    then run via terminal. The only exception is quick one-off calculations with no output file.
    After writing a file, use the terminal tool to run python3 -m py_compile <file>
    to check syntax, then run it with python3 <file>. Never use subprocess inside this
    tool to run other scripts — always use the terminal tool for that.
    Use this tool to write all multi-line files — never use echo in terminal for file writing.

    Args:
        code: Valid Python source code to execute.

    Returns:
        Combined stdout and stderr output from the execution.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 60 seconds."
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


_terminal_cwd: str | None = None

_CWD_SENTINEL = "__TERMINAL_CWD__:"

_DESTRUCTIVE_PATTERNS = [
    r"\brm\b.*-[a-z]*r",       # rm -r, rm -rf, rm -fr, etc.
    r"\brm\b.*\*",             # rm with wildcard
    r"\brm\b.*/\S",            # rm with absolute path
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[a-z]*f",
    r"find\b.*-delete",        # find -delete
    r"find\b.*-exec\s+rm",     # find -exec rm
]

def _is_destructive(command: str) -> bool:
    low = command.lower()
    return any(re.search(p, low) for p in _DESTRUCTIVE_PATTERNS)


_auto_approved: set[str] = set()


def _base_cmd(command: str) -> str:
    parts = command.strip().split()
    for part in parts:
        if "=" not in part:
            return part
    return parts[0] if parts else command


async def terminal(command: str) -> str:
    """Run a shell command and stream output line by line.

    CWD persists across calls — cd once and it stays for future calls.
    Before entering a directory the user mentions by name, run ls -F first to
    confirm it exists — never assume or create it.
    Never run ls -R (node_modules overflow) — use ls -F or find -maxdepth 2.
    Never use echo, printf, or heredoc to write file contents — use the python
    tool with open('/abs/path', 'w').write(content) for all file creation.
    Never start or run a server, daemon, or any long-running process — write
    the file, then tell the user the exact command to run and the URL. Never
    run flask run, uvicorn, node server.js, npm start, or similar.
    Install packages with: python3 -m pip install X (pip is not on PATH).
    Syntax check: python3 -m py_compile <file> — never claim correct without running.
    Pass --yes/-y/--force to skip interactive prompts.

    Args:
        command: The shell command to execute (runs via bash).

    Returns:
        Combined stdout and stderr from the command.
    """
    global _terminal_cwd
    from rich.live import Live
    from rich.text import Text
    from ui import console, render_terminal_live

    base = _base_cmd(command)
    if base not in _auto_approved:
        from ui import confirm_terminal
        answer = await confirm_terminal(command, base, is_destructive=_is_destructive(command))
        if answer in ("r", "remember"):
            _auto_approved.add(base)
        elif not answer.startswith("y"):
            return "Command cancelled by user."

    if _terminal_cwd:
        full_command = f"cd {_terminal_cwd!r} && {command}; echo '{_CWD_SENTINEL}'\"$(pwd)\""
    else:
        full_command = f"{command}; echo '{_CWD_SENTINEL}'\"$(pwd)\""

    proc = await asyncio.create_subprocess_shell(
        full_command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        executable="/bin/bash",
    )

    output_parts: list[str] = []
    _CONFIRM_PATTERNS = ("(y/n)", "[y/n]", "(yes/no)", "ok to proceed", "? (y)", ": (y)")

    interrupted = False
    timed_out = False

    with Live(render_terminal_live([], done=False), console=console, refresh_per_second=10) as live:

        async def _read_stream() -> None:
            async for line in proc.stdout:
                text = line.decode(errors="replace")
                output_parts.append(text)
                live.update(render_terminal_live(output_parts, done=False))
                low = text.strip().lower()
                if any(p in low for p in _CONFIRM_PATTERNS):
                    proc.stdin.write(b"y\n")
                    await proc.stdin.drain()

        read_task = asyncio.ensure_future(_read_stream())
        try:
            await asyncio.wait_for(asyncio.shield(read_task), timeout=60)
        except asyncio.TimeoutError:
            timed_out = True
            read_task.cancel()
            proc.kill()
            await proc.wait()
        except asyncio.CancelledError:
            interrupted = True
            read_task.cancel()
            proc.send_signal(__import__("signal").SIGINT)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        live.update(render_terminal_live(output_parts, done=True))

    await proc.wait()
    raw = "".join(output_parts)

    # Extract and strip the CWD sentinel line
    lines = raw.splitlines()
    clean_lines = []
    for line in lines:
        if line.startswith(_CWD_SENTINEL):
            new_cwd = line[len(_CWD_SENTINEL):].strip()
            if new_cwd:
                _terminal_cwd = new_cwd
        else:
            clean_lines.append(line)
    result = "\n".join(clean_lines).strip()

    if interrupted:
        return (result + "\n(interrupted by user)").strip()
    if timed_out:
        return (result + "\nError: command timed out after 1 minute.").strip()
    return result or "(no output)"
