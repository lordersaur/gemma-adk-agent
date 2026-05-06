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
    """Fetch a URL and return cleaned readable text."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
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
            print(f"  [web_search] reading {url}")
            content = _fetch_page(url)
            sections.append(
                f"[{i}] {r['title']}\n{url}\n\n{content}"
            )
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


def terminal(command: str) -> str:
    """Run a shell command and return its output.

    Args:
        command: The shell command to execute (runs via bash).

    Returns:
        Combined stdout and stderr from the command.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
