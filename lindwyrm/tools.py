"""Tools the agent can call, plus their Anthropic tool schemas.

Each tool is a function (cfg, **input) -> str. The returned string is fed back
to the model as the tool_result. Errors are caught by the caller and returned
as tool_result with is_error=True.

Confirmation and sandboxing happen INSIDE each tool, so the policy is enforced
no matter how the model phrases its request.
"""

from __future__ import annotations

import fnmatch
import os
import signal
import subprocess
from pathlib import Path

from .config import Config
from .offload import get_store
from .sandbox import (
    SandboxError,
    UserQuit,
    authorize,
    bash_confirm,
    resolve_target,
)

MAX_READ_BYTES = 256 * 1024  # don't dump huge files into context blindly
MAX_GREP_BYTES = 8 * 1024 * 1024  # skip files too big to be worth searching


def _looks_binary(path: Path) -> bool:
    """Cheap binary sniff: a NUL byte in the first 4 KiB."""
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(4096)
    except OSError:
        return True

# Directories that are almost never what a search is looking for. Walking them
# wastes time and floods the model's context with vendored or generated code.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components", "vendor",
    ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".nox", ".eggs",
    "dist", "build", "target", ".next", ".nuxt", ".output",
    ".idea", ".vscode", ".gradle", ".terraform", ".cache",
})


def _is_skipped(path: Path, base: Path) -> bool:
    """True if `path` sits inside one of SKIP_DIRS, relative to `base`."""
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        parts = path.parts
    return any(p in SKIP_DIRS for p in parts)


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a text file. Returns the file content with "
            "1-based line numbers prefixed. Use this before editing a file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "start_line": {"type": "integer", "description": "Optional 1-based start line."},
                "end_line": {"type": "integer", "description": "Optional 1-based end line (inclusive)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file or overwrite an existing one with the given "
            "content. Creates parent directories if needed. Requires write "
            "permission and the path must be inside an allowed write root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact substring in a file with new text. By default "
            "`old_text` must appear EXACTLY once; if it appears several times "
            "the error tells you which lines matched, so you can either add "
            "surrounding context or set replace_all. Prefer this over "
            "write_file for small changes so you don't rewrite the whole file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to replace."},
                "new_text": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and subdirectories in a directory (one level).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path. Defaults to project root."}},
            "required": [],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern (recursive), e.g. '**/*.py'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Base directory. Defaults to project root."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search file contents for a literal substring or regex. Returns matching lines with file:line prefixes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Base directory. Defaults to project root."},
                "glob": {"type": "string", "description": "Optional file filter, e.g. '*.py'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file. Requires delete permission and write-root membership.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_offloaded",
        "description": (
            "Retrieve a tool result that was moved out of the conversation to "
            "save context. Use the ref from an [offloaded: ...] marker. This "
            "returns a SNAPSHOT taken when the original tool ran -- if you "
            "want a file's current contents instead, use read_file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The ref, e.g. 'off_0003'."},
                "start_line": {"type": "integer", "description": "Optional 1-based start line."},
                "end_line": {"type": "integer", "description": "Optional 1-based end line (inclusive)."},
            },
            "required": ["ref"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command in the project root and return its combined "
            "stdout/stderr. Requires bash permission. Has a timeout. Use for "
            "running tests, git, build commands, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds (default 60, max 600)."},
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def _rel(cfg: Config, p: Path) -> str:
    try:
        return str(p.relative_to(cfg.project_root))
    except ValueError:
        return str(p)


def _reld(cfg: Config, raw_path: str) -> str:
    """Relative display string for a raw (possibly relative) path argument."""
    return _rel(cfg, resolve_target(cfg, raw_path))


def tool_read_file(cfg: Config, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    target = authorize(cfg, "read", path, f"read {_reld(cfg, path)}")
    if not target.is_file():
        raise SandboxError(f"Not a file: {target}")
    if _looks_binary(target):
        size = target.stat().st_size
        raise SandboxError(
            f"{_rel(cfg, target)} looks like a binary file ({size} bytes). "
            f"Reading it would put mojibake in the conversation; use bash "
            f"(file, xxd, strings) if you need to inspect it."
        )
    try:
        data = target.read_bytes()
    except OSError as e:
        raise SandboxError(f"Could not read {_rel(cfg, target)}: {e}") from None
    if len(data) > MAX_READ_BYTES and start_line is None:
        raise SandboxError(
            f"File is {len(data)} bytes (> {MAX_READ_BYTES}). "
            "Read a line range with start_line/end_line instead."
        )
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return "(empty file)"

    s = max(0, (start_line - 1) if start_line else 0)
    e = min(len(lines), end_line if end_line else len(lines))
    if s >= len(lines):
        # Saying "(empty file)" here was a plain lie, and the model believed
        # it -- an out-of-range window is not an empty file.
        raise SandboxError(
            f"start_line {start_line} is past the end of "
            f"{_rel(cfg, target)}, which has {len(lines)} lines."
        )
    if e <= s:
        raise SandboxError(
            f"Empty line range: start_line {start_line} is not before "
            f"end_line {end_line}."
        )
    width = len(str(e))
    return "\n".join(f"{str(i + 1).rjust(width)}\t{lines[i]}" for i in range(s, e))


def tool_write_file(cfg: Config, path: str, content: str) -> str:
    resolved = resolve_target(cfg, path)
    exists = resolved.is_file()
    verb = "overwrite" if exists else "create"
    preview = content if len(content) <= 800 else content[:800] + f"\n... ({len(content)} chars total)"
    target = authorize(cfg, "write", path,
                       f"{verb} {_rel(cfg, resolved)} ({len(content)} chars)", preview=preview)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {_rel(cfg, target)}"


def _match_lines(text: str, needle: str) -> list[int]:
    """1-based line numbers where each occurrence of `needle` starts."""
    lines = []
    start = 0
    while (idx := text.find(needle, start)) != -1:
        lines.append(text.count("\n", 0, idx) + 1)
        start = idx + max(1, len(needle))
    return lines


def tool_edit_file(cfg: Config, path: str, old_text: str, new_text: str,
                   replace_all: bool = False) -> str:
    resolved = resolve_target(cfg, path)
    if not resolved.is_file():
        raise SandboxError(f"Not a file: {resolved}")
    original = resolved.read_text(encoding="utf-8")
    count = original.count(old_text)
    if count == 0:
        raise SandboxError("old_text not found in file.")
    if count > 1 and not replace_all:
        # Point at every match so the next attempt can add the right context
        # instead of guessing -- repeated boilerplate is where this used to
        # stall, with nothing to go on but "it must be unique".
        where = ", ".join(f"line {n}" for n in _match_lines(original, old_text)[:10])
        raise SandboxError(
            f"old_text appears {count} times ({where}). Either extend it with "
            f"surrounding context so it matches exactly once, or pass "
            f"replace_all=true to change every occurrence."
        )

    if replace_all:
        updated = original.replace(old_text, new_text)
        summary = f"edit {_rel(cfg, resolved)} ({count} occurrence(s))"
    else:
        updated = original.replace(old_text, new_text, 1)
        summary = f"edit {_rel(cfg, resolved)}"

    diff_preview = _mini_diff(old_text, new_text)
    target = authorize(cfg, "write", path, summary, preview=diff_preview)
    target.write_text(updated, encoding="utf-8")
    return (f"Edited {_rel(cfg, target)}"
            + (f" ({count} occurrences replaced)" if replace_all else ""))


def _mini_diff(old: str, new: str) -> str:
    out = []
    for line in old.splitlines():
        out.append(f"- {line}")
    for line in new.splitlines():
        out.append(f"+ {line}")
    return "\n".join(out)


def tool_list_dir(cfg: Config, path: str | None = None) -> str:
    raw = path or str(cfg.project_root)
    target = authorize(cfg, "read", raw, f"list {_reld(cfg, raw)}")
    if not target.exists():
        raise SandboxError(f"Does not exist: {target}")
    if not target.is_dir():
        raise SandboxError(f"Not a directory: {target}")
    try:
        entries = sorted(target.iterdir(),
                         key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        raise SandboxError(f"Could not list {_rel(cfg, target)}: {e}") from None

    out = []
    for p in entries:
        # Dotfiles are listed. Hiding them meant .github/, .gitignore and
        # every config file were invisible -- and silently so, since glob
        # returns dotfiles but never directories, leaving a hidden directory
        # unreachable by any read tool.
        if p.is_symlink():
            try:
                out.append(f"{p.name}@ -> {os.readlink(p)}")
            except OSError:
                out.append(f"{p.name}@")
        elif p.is_dir():
            out.append(f"{p.name}/")
        else:
            out.append(p.name)
    return "\n".join(out) if out else "(empty directory)"


def tool_glob(cfg: Config, pattern: str, path: str | None = None) -> str:
    raw = path or str(cfg.project_root)
    base = authorize(cfg, "read", raw, f"glob {pattern} in {_reld(cfg, raw)}")
    if base.is_file():
        raise SandboxError(
            f"{_rel(cfg, base)} is a file, not a directory. glob searches "
            f"inside a directory -- pass the containing folder, or use grep "
            f"to search this file's contents."
        )
    matches = sorted(
        str(p.relative_to(base))
        for p in base.glob(pattern)
        if p.is_file() and not _is_skipped(p, base)
    )
    if len(matches) > 500:
        return "\n".join(matches[:500]) + f"\n... ({len(matches)} matches, showing 500)"
    return "\n".join(matches) if matches else "(no matches)"


def tool_grep(cfg: Config, pattern: str, path: str | None = None, glob: str | None = None) -> str:
    import re

    raw = path or str(cfg.project_root)
    base = authorize(cfg, "read", raw, f"grep {pattern!r} in {_reld(cfg, raw)}")
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))
    results = []
    if base.is_file():
        # A file here used to yield nothing at all, because rglob on a file is
        # empty -- so grep answered "(no matches)" for a file full of matches.
        # A confident wrong answer is worse than an error: the model believes
        # it and goes looking somewhere else.
        files = [base]
        base = base.parent
    else:
        files = base.rglob(glob) if glob else base.rglob("*")
    for f in files:
        if not f.is_file() or _is_skipped(f, base):
            continue
        try:
            if f.stat().st_size > MAX_GREP_BYTES or _looks_binary(f):
                continue
            # Streamed line by line: reading whole files in was fine for source
            # but pulls a multi-hundred-megabyte log entirely into memory.
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        results.append(f"{f.relative_to(base)}:{i}: {line.strip()[:200]}")
                        if len(results) >= 300:
                            return "\n".join(results) + "\n... (truncated)"
        except OSError:
            continue
    return "\n".join(results) if results else "(no matches)"


def tool_delete_file(cfg: Config, path: str) -> str:
    resolved = resolve_target(cfg, path)
    # A dangling symlink exists as a link but not as a target, and exists()
    # follows the link -- so a broken link could never be removed.
    raw = Path(os.path.expanduser(path))
    if not raw.is_absolute():
        raw = cfg.project_root / raw
    if raw.is_symlink():
        target = authorize(cfg, "delete", path, f"DELETE symlink {_rel(cfg, raw)}")
        raw.unlink()
        return f"Deleted symlink {_rel(cfg, raw)} (its target was left alone)"
    if not resolved.exists():
        raise SandboxError(f"Does not exist: {resolved}")
    if resolved.is_dir():
        raise SandboxError("Refusing to delete a directory; only files are supported.")
    target = authorize(cfg, "delete", path, f"DELETE {_rel(cfg, resolved)}")
    target.unlink()
    return f"Deleted {_rel(cfg, target)}"


def tool_read_offloaded(cfg: Config, ref: str, start_line: int | None = None,
                        end_line: int | None = None) -> str:
    """Read back offloaded content. Not path-scoped: this is lindwyrm's own
    session data, addressed by ref, not a file the user asked us to guard."""
    try:
        return get_store().get(ref, start_line, end_line)
    except KeyError as e:
        raise SandboxError(str(e)) from None
    except OSError as e:
        raise SandboxError(f"Could not read offloaded content: {e}") from None


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """SIGTERM the command's whole process group, SIGKILL what survives."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue


def tool_bash(cfg: Config, command: str, timeout: int = 60) -> str:
    if cfg.policy.read_only:
        raise SandboxError("Bash blocked: running in read-only mode.")
    cmd = command.strip()
    # Hard denylist always applies.
    for bad in cfg.policy.bash_denylist:
        if bad in cmd:
            raise SandboxError(f"Bash blocked: command matches denylist entry {bad!r}.")
    # Allowlist shortcut.
    perm = cfg.policy.bash
    if any(cmd.startswith(a) for a in cfg.policy.bash_allowlist):
        perm = "allow"
    if not bash_confirm(perm, f"run: {cmd}"):
        raise SandboxError("Bash declined by user.")
    timeout = max(1, min(int(timeout), 600))
    # start_new_session puts the command in its own process group so that a
    # timeout or Ctrl+C kills the whole tree. subprocess.run() would only
    # signal the shell itself, leaving its children running in the background.
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(cfg.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        proc.communicate()
        raise SandboxError(f"Command timed out after {timeout}s.")
    except KeyboardInterrupt:
        _kill_process_tree(proc)
        proc.communicate()
        raise
    out = (stdout or "") + (stderr or "")
    out = out.strip() or "(no output)"
    if len(out) > 30_000:
        out = out[:30_000] + "\n... (output truncated)"
    return f"exit code: {proc.returncode}\n{out}"


# Dispatch table.
TOOL_FUNCS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "glob": tool_glob,
    "grep": tool_grep,
    "delete_file": tool_delete_file,
    "read_offloaded": tool_read_offloaded,
    "bash": tool_bash,
}


def run_tool(cfg: Config, name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool by name. Returns (result_text, is_error)."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Unknown tool: {name}", True
    try:
        result = func(cfg, **tool_input)
        return result, False
    except (UserQuit, KeyboardInterrupt):
        # Leaving the session is the user's decision, not a tool failure --
        # it must reach the REPL rather than be reported back to the model.
        raise
    except SandboxError as e:
        return str(e), True
    except TypeError as e:
        return f"Bad tool input for {name}: {e}", True
    except Exception as e:  # noqa: BLE001
        return f"Error running {name}: {type(e).__name__}: {e}", True
