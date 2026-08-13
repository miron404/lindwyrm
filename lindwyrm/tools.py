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
from .sandbox import (
    SandboxError,
    UserQuit,
    authorize,
    bash_confirm,
    resolve_target,
)

MAX_READ_BYTES = 256 * 1024  # don't dump huge files into context blindly

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
            "Replace an exact substring in a file with new text. `old_text` "
            "must appear EXACTLY once in the file. Prefer this over write_file "
            "for small changes so you don't rewrite the whole file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to replace (must be unique)."},
                "new_text": {"type": "string", "description": "Replacement text."},
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
    data = target.read_bytes()
    if len(data) > MAX_READ_BYTES and start_line is None:
        raise SandboxError(
            f"File is {len(data)} bytes (> {MAX_READ_BYTES}). "
            "Read a line range with start_line/end_line instead."
        )
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    s = (start_line - 1) if start_line else 0
    e = end_line if end_line else len(lines)
    s = max(0, s)
    e = min(len(lines), e)
    width = len(str(e))
    out = [f"{str(i + 1).rjust(width)}\t{lines[i]}" for i in range(s, e)]
    return "\n".join(out) if out else "(empty file)"


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


def tool_edit_file(cfg: Config, path: str, old_text: str, new_text: str) -> str:
    resolved = resolve_target(cfg, path)
    if not resolved.is_file():
        raise SandboxError(f"Not a file: {resolved}")
    original = resolved.read_text(encoding="utf-8")
    count = original.count(old_text)
    if count == 0:
        raise SandboxError("old_text not found in file.")
    if count > 1:
        raise SandboxError(f"old_text appears {count} times; it must be unique. Add more context.")
    updated = original.replace(old_text, new_text, 1)
    diff_preview = _mini_diff(old_text, new_text)
    target = authorize(cfg, "write", path, f"edit {_rel(cfg, resolved)}", preview=diff_preview)
    target.write_text(updated, encoding="utf-8")
    return f"Edited {_rel(cfg, target)}"


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
    if not target.is_dir():
        raise SandboxError(f"Not a directory: {target}")
    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    out = []
    for p in entries:
        if p.name.startswith("."):
            continue
        out.append(f"{p.name}/" if p.is_dir() else p.name)
    return "\n".join(out) if out else "(empty directory)"


def tool_glob(cfg: Config, pattern: str, path: str | None = None) -> str:
    raw = path or str(cfg.project_root)
    base = authorize(cfg, "read", raw, f"glob {pattern} in {_reld(cfg, raw)}")
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
    files = base.rglob(glob) if glob else base.rglob("*")
    for f in files:
        if not f.is_file() or _is_skipped(f, base):
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    results.append(f"{f.relative_to(base)}:{i}: {line.strip()[:200]}")
                    if len(results) >= 300:
                        return "\n".join(results) + "\n... (truncated)"
        except OSError:
            continue
    return "\n".join(results) if results else "(no matches)"


def tool_delete_file(cfg: Config, path: str) -> str:
    resolved = resolve_target(cfg, path)
    if not resolved.exists():
        raise SandboxError(f"Does not exist: {resolved}")
    if resolved.is_dir():
        raise SandboxError("Refusing to delete a directory; only files are supported.")
    target = authorize(cfg, "delete", path, f"DELETE {_rel(cfg, resolved)}")
    target.unlink()
    return f"Deleted {_rel(cfg, target)}"


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
