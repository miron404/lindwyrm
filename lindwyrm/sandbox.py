"""Permission resolution and interactive confirmation.

Every filesystem operation goes through authorize(), which:
  1. Resolves the target to an absolute, symlink-resolved path (so `../` and
     symlink tricks can't dodge a rule).
  2. Asks the Policy for the effective level (allow / confirm / deny) for that
     operation on that path, honoring per-path rules by specificity.
  3. allow -> proceed silently; deny -> blocked; confirm -> prompt the user.

bash is handled separately in tools.py since it isn't path-bound.
"""

from __future__ import annotations

import os
from pathlib import Path


class SandboxError(Exception):
    """Raised when an operation is blocked by policy."""


class UserQuit(Exception):
    """Raised when the user answers [q]uit at a confirmation prompt.

    Distinct from KeyboardInterrupt on purpose: answering "quit" is a
    deliberate choice, while Ctrl+C is an interrupt, and any `except
    KeyboardInterrupt` up the stack would otherwise swallow the two
    identically.
    """


def resolve_target(cfg, path: str) -> Path:
    """Turn a user/model-supplied path into an absolute resolved Path.

    Relative paths are taken against the project root. Symlinks are resolved so
    a rule on a real directory can't be bypassed via a link.
    """
    target = Path(os.path.expanduser(path))
    if not target.is_absolute():
        target = cfg.project_root / target
    try:
        return target.resolve()
    except (OSError, RuntimeError):
        return target


# ---------------------------------------------------------------------------
# Confirmation handling
# ---------------------------------------------------------------------------

# "always allow this op for the rest of the turn" grants, keyed by operation.
_session_grants: set[str] = set()


def reset_session_grants() -> None:
    _session_grants.clear()


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "n"


def authorize(
    cfg,
    op: str,
    path: str,
    summary: str,
    *,
    preview: str | None = None,
) -> Path:
    """Authorize an operation on a path. Returns the resolved Path or raises.

    op is one of "read" | "write" | "delete". Resolves the effective permission
    from policy (global default + per-path rules) and prompts if needed.
    """
    target = resolve_target(cfg, path)
    level = cfg.policy.effective(op, target)

    if level == "deny":
        raise SandboxError(f"{op.capitalize()} denied by policy for {target}.")
    if level == "allow":
        return target

    # confirm
    if op in _session_grants:
        return target

    print()
    print(f"  \033[1;33m{op.upper()} requested:\033[0m {summary}")
    if preview:
        for line in preview.splitlines():
            print(f"    \033[2m{line}\033[0m")
    ans = _ask(f"  Allow? [y]es / [n]o / [a]lways ({op}) / [q]uit: ")
    if ans in ("y", "yes"):
        return target
    if ans in ("a", "always"):
        _session_grants.add(op)
        return target
    if ans in ("q", "quit"):
        raise UserQuit
    raise SandboxError(f"{op.capitalize()} declined by user.")


def bash_confirm(permission: str, summary: str) -> bool:
    """Confirm a bash command (path-independent). Returns True to proceed."""
    if permission == "deny":
        print(f"  [denied by policy] {summary}")
        return False
    if permission == "allow":
        return True
    if "bash" in _session_grants:
        return True
    print()
    print(f"  \033[1;33mBASH requested:\033[0m {summary}")
    ans = _ask("  Allow? [y]es / [n]o / [a]lways (bash) / [q]uit: ")
    if ans in ("y", "yes"):
        return True
    if ans in ("a", "always"):
        _session_grants.add("bash")
        return True
    if ans in ("q", "quit"):
        raise UserQuit
    return False
