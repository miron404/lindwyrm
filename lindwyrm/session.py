"""Saving and resuming conversations.

Closing the terminal used to throw away everything: the reasoning, the files
already read, the decisions already made. A session is written to disk after
every turn, so the work survives a closed laptop, a crash, or a Ctrl+D.

One JSON file per session under the user's data directory. Sessions are
scoped to the project they ran in, so `--continue` in one checkout never
resumes work from another.

These files hold whatever the agent saw: source code, command output,
sometimes the contents of a config file it was asked to read. They are
written 0600 in a 0700 directory rather than left at whatever the umask
happens to be.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

SESSIONS_ROOT = Path(os.path.expanduser("~/.local/share/lindwyrm/sessions"))
RETENTION_DAYS = 30
TITLE_CHARS = 60


def new_session_id() -> str:
    """Sortable id: newest last alphabetically, which keeps listing simple."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{random.randint(0x1000, 0xffff):04x}"


def session_path(session_id: str, root: Path | None = None) -> Path:
    return (root or SESSIONS_ROOT) / f"{session_id}.json"


def _title_from(messages: list[dict]) -> str:
    """First thing the user actually asked, for the listing."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "text":
                text = " ".join(block.get("text", "").split())
                if text.startswith("[Summary of earlier conversation"):
                    continue  # a compaction artefact, not something they typed
                if text:
                    return text[:TITLE_CHARS]
    return "(no prompt yet)"


def save_session(session_id: str, state: dict, root: Path | None = None) -> Path | None:
    """Write the session atomically. Returns the path, or None on failure.

    Never raises: losing a save is bad, but killing a working session over it
    would be worse.
    """
    path = session_path(session_id, root)
    state = dict(state)
    state["id"] = session_id
    state["updated"] = time.time()
    state.setdefault("created", state["updated"])
    state["title"] = _title_from(state.get("messages", []))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
    except OSError:
        pass

    tmp = path.with_suffix(".tmp")
    try:
        # Written to a temp file and renamed: a crash mid-write would
        # otherwise leave a truncated file where a session used to be.
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    return path


def load_session(session_id: str, root: Path | None = None) -> dict | None:
    path = session_path(session_id, root)
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def list_sessions(project_root: str | Path | None = None, *, limit: int = 20,
                  root: Path | None = None) -> list[dict]:
    """Recent sessions, newest first, optionally only for one project."""
    directory = root or SESSIONS_ROOT
    if not directory.is_dir():
        return []
    wanted = str(Path(project_root).resolve()) if project_root else None

    found: list[dict] = []
    try:
        entries = sorted(directory.glob("*.json"), reverse=True)
    except OSError:
        return []
    for path in entries:
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # a half-written or hand-mangled file shouldn't stop the list
        if wanted and data.get("project_root") != wanted:
            continue
        found.append({
            "id": data.get("id", path.stem),
            "title": data.get("title", ""),
            "updated": data.get("updated", 0),
            "messages": len(data.get("messages", [])),
            "preset": data.get("preset", "?"),
            "project_root": data.get("project_root", ""),
        })
        if len(found) >= limit:
            break
    return found


def latest_session_id(project_root: str | Path, root: Path | None = None) -> str | None:
    sessions = list_sessions(project_root, limit=1, root=root)
    return sessions[0]["id"] if sessions else None


def sweep_sessions(max_age_days: int = RETENTION_DAYS,
                   root: Path | None = None) -> int:
    """Delete sessions older than the retention window."""
    directory = root or SESSIONS_ROOT
    if not directory.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        for path in directory.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def describe_age(updated: float) -> str:
    """Human-friendly age for the listing."""
    seconds = max(0, time.time() - updated)
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m ago"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"
