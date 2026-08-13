"""Offloading bulky tool results out of the conversation and onto disk.

The cheapest way to reclaim context is not to summarize it -- it is to move
the bulk somewhere retrievable and leave a stub behind. Summarizing is lossy
and irreversible; offloading is neither, because the full text stays on disk
and the model can fetch it back with read_offloaded.

Why a snapshot rather than "just read the file again": the file may have been
edited or deleted since the tool ran. Re-reading would then answer a different
question than the one the transcript records, and the model ends up reasoning
against its own history -- "I read this and it said A" versus a file that now
says B. The stub is explicit that it holds a point-in-time copy, and points at
read_file for whatever the file looks like now.

Layout: one directory per session under the user's data dir, one plain text
file per offloaded result, so the content stays greppable from outside. Stale
session directories are swept on startup -- a crash shouldn't leak disk space
forever.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

CHARS_PER_TOKEN = 4  # rough, and only used for "is this big enough to bother"
PREVIEW_LINES = 8
PREVIEW_CHARS = 400  # a minified file is one enormous line; cap by size too
SESSION_RETENTION_DAYS = 7

DEFAULT_ROOT = Path(os.path.expanduser("~/.local/share/lindwyrm/offload"))


def estimate_text_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


@dataclass
class OffloadEntry:
    ref: str
    path: Path
    label: str  # e.g. "read_file src/parser.py"
    lines: int
    chars: int


class OffloadStore:
    """Holds offloaded tool results for one session."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (DEFAULT_ROOT / f"{int(time.time())}-{os.getpid()}")
        self._entries: dict[str, OffloadEntry] = {}
        self._counter = 0
        self._ready = False

    # -- storage ------------------------------------------------------------

    def _ensure_root(self) -> None:
        if not self._ready:
            self.root.mkdir(parents=True, exist_ok=True)
            self._ready = True

    def put(self, text: str, label: str) -> OffloadEntry | None:
        """Write `text` to disk and return its entry, or None if that failed.

        A failure here must never break the turn: the caller keeps the result
        inline instead, which costs context but loses nothing.
        """
        try:
            self._ensure_root()
            self._counter += 1
            ref = f"off_{self._counter:04d}"
            path = self.root / f"{ref}.txt"
            path.write_text(text, encoding="utf-8")
        except OSError:
            return None
        entry = OffloadEntry(ref=ref, path=path, label=label,
                             lines=text.count("\n") + 1, chars=len(text))
        self._entries[ref] = entry
        return entry

    def get(self, ref: str, start_line: int | None = None,
            end_line: int | None = None) -> str:
        """Read back an offloaded result. Raises KeyError for unknown refs."""
        entry = self._entries.get(ref.strip())
        if entry is None:
            known = ", ".join(sorted(self._entries)) or "none yet"
            raise KeyError(f"unknown offload ref {ref!r}; available: {known}")
        text = entry.path.read_text(encoding="utf-8", errors="replace")
        if start_line is None and end_line is None:
            return text
        lines = text.splitlines()
        start = max(0, (start_line - 1) if start_line else 0)
        end = min(len(lines), end_line if end_line else len(lines))
        return "\n".join(lines[start:end])

    def __len__(self) -> int:
        return len(self._entries)

    def total_chars(self) -> int:
        return sum(e.chars for e in self._entries.values())

    # -- the text left behind in the conversation ---------------------------

    def stub(self, entry: OffloadEntry, text: str) -> str:
        preview = "\n".join(text.splitlines()[:PREVIEW_LINES])
        if len(preview) > PREVIEW_CHARS:
            # One 4000-character line counts as a single "line", so a
            # line-only limit can quote the whole result back and make the
            # stub larger than what it replaces.
            preview = preview[:PREVIEW_CHARS] + " …"
        hidden = max(0, entry.lines - PREVIEW_LINES)
        size_kb = entry.chars / 1024
        return (
            f"[offloaded: {entry.label} -- {entry.lines} lines, {size_kb:.1f} KB, "
            f"ref {entry.ref}]\n"
            f"{preview}\n"
            f"... {hidden} more line(s) not shown.\n"
            f"This is a snapshot from when the tool ran; the file or command "
            f"output may differ now. Use read_offloaded(\"{entry.ref}\") for the "
            f"full snapshot, or read_file for the file's current contents."
        )

    def cleanup(self) -> None:
        """Remove this session's directory."""
        try:
            if self._ready and self.root.exists():
                shutil.rmtree(self.root, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Session-wide store, mirroring how sandbox.py keeps its session grants.
# ---------------------------------------------------------------------------

_store: OffloadStore | None = None


def get_store() -> OffloadStore:
    global _store
    if _store is None:
        _store = OffloadStore()
        sweep_old_sessions()
    return _store


def set_store(store: OffloadStore | None) -> None:
    """Replace the session store (used by tests)."""
    global _store
    _store = store


def sweep_old_sessions(root: Path | None = None,
                       max_age_days: int = SESSION_RETENTION_DAYS) -> int:
    """Delete offload directories left behind by crashed sessions."""
    root = root or DEFAULT_ROOT
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        return removed
    return removed
