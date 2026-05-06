import asyncio
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


def _fetch_page(url: str, max_chars: int = 4000) -> str:
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
    """Search the web and read the content of the top results.

    Args:
        query: The search query string.
        max_results: Number of pages to search and read (default 3).

    Returns:
        Titles, URLs, and full page content for each result.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."

        sections = []
        for i, r in enumerate(results, 1):
            url = r["href"]
            content = _fetch_page(url)
            sections.append(f"[{i}] {r['title']}\n{url}\n\n{content}")
        return "\n\n---\n\n".join(sections)
    except Exception as e:
        return f"Search error: {e}"


def python(code: str) -> str:
    """Execute Python code and return stdout + stderr.

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
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


_terminal_cwd: str | None = None

_CWD_SENTINEL = "__TERMINAL_CWD__:"


async def terminal(command: str) -> str:
    """Run a shell command and stream output line by line.

    The working directory persists across calls — cd commands carry over to the
    next tool call just like a real shell session.
    For interactive CLI tools that prompt for input, pass flags to skip prompts
    (e.g. --yes, -y, --force).
    Long-running commands like package installs are allowed up to 3 minutes.

    Args:
        command: The shell command to execute (runs via bash).

    Returns:
        Combined stdout and stderr from the command.
    """
    global _terminal_cwd
    from rich.live import Live
    from ui import console, render_terminal_live

    if _terminal_cwd:
        full_command = f"cd {_terminal_cwd!r} && ( {command} ); echo '{_CWD_SENTINEL}'\"$(pwd)\""
    else:
        full_command = f"( {command} ); echo '{_CWD_SENTINEL}'\"$(pwd)\""

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
            await asyncio.wait_for(asyncio.shield(read_task), timeout=180)
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
        return (result + "\nError: command timed out after 3 minutes.").strip()
    return result or "(no output)"
