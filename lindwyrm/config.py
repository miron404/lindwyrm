"""Configuration loading, presets, and security policy for lindwyrm.

Settings come from (highest priority first):
  1. CLI flags
  2. A project-local config file: ./.lindwyrm.toml
  3. A user config file: ~/.config/lindwyrm/config.toml
  4. Built-in defaults

API keys are read ONLY from the environment or from a key file referenced in
config -- never stored inline in a committed config, to avoid leaking them
into a repo.

Presets bundle everything needed to talk to one provider/model: the wire
format ("anthropic" or "openai"), base_url, model id, and which env var(s)
hold the key. DeepSeek Flash/Pro ship built in; add your own via [[presets]]
in config to point at any other Anthropic- or OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

try:  # py311+ has tomllib in stdlib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


# Permission levels.
#   allow   -> run without asking
#   confirm -> ask the user every time
#   deny    -> never run (alias: forbid)
Permission = Literal["allow", "confirm", "deny"]


# Accept "forbid" as a synonym for "deny" in configs/commands.
def normalize_permission(val: str) -> str:
    v = val.strip().lower()
    if v == "forbid":
        return "deny"
    return v


DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
ANTHROPIC_VERSION = "2023-06-01"

# Max tokens ceiling for the DeepSeek V4 series (thinking tokens count toward
# this). Other providers may have their own, lower ceilings; they'll reject an
# over-large value themselves, so this is just a generous sanity cap.
MODEL_MAX_TOKENS_CEILING = 384_000


# ---------------------------------------------------------------------------
# Presets: everything needed to talk to one provider/model.
# ---------------------------------------------------------------------------

@dataclass
class Preset:
    name: str
    format: str = "anthropic"  # "anthropic" | "openai"
    base_url: str = DEFAULT_BASE_URL
    model: str = "deepseek-v4-flash"
    # Env var names to try, in order, for the API key.
    api_key_env: tuple[str, ...] = ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")
    key_file: str | None = None  # fallback if no env var is set
    thinking: bool = True
    max_tokens: int = 8192
    thinking_budget: int = 4096
    temperature: float | None = None
    # Size of the model's context window, in tokens. Used to decide when the
    # conversation should be compacted -- it is not sent to the API.
    context_limit: int = 128_000
    # Send max_completion_tokens instead of max_tokens (OpenAI format only).
    max_completion_tokens: bool = False
    # Extra top-level fields merged into every request body verbatim (e.g. a
    # provider-specific flag). Rarely needed.
    extra_body: dict = field(default_factory=dict)


BUILTIN_PRESETS: dict[str, Preset] = {
    "deepseek-flash": Preset(
        name="deepseek-flash", format="anthropic",
        base_url=DEFAULT_BASE_URL, model="deepseek-v4-flash",
    ),
    "deepseek-pro": Preset(
        name="deepseek-pro", format="anthropic",
        base_url=DEFAULT_BASE_URL, model="deepseek-v4-pro",
    ),
}
PRESET_ALIASES = {"flash": "deepseek-flash", "pro": "deepseek-pro"}
DEFAULT_PRESET = "deepseek-flash"


def _resolve_preset_api_key(preset: Preset, global_key_file: str | None) -> str:
    for env_name in preset.api_key_env:
        val = os.environ.get(env_name)
        if val:
            return val.strip()
    key_file = preset.key_file or global_key_file
    if key_file:
        p = Path(os.path.expanduser(key_file))
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    tried = ", ".join(preset.api_key_env) if preset.api_key_env else "(none configured)"
    raise SystemExit(
        f"No API key found for preset '{preset.name}'. Set one of: {tried}, "
        f"or set key_file (globally or on this preset in config)."
    )


def _build_presets(data: dict) -> dict[str, Preset]:
    """Builtin presets plus any user-defined [[presets]] entries.

    A user entry with the same `name` as a builtin overrides it; fields you
    don't set are inherited from the builtin/previous definition of that name.
    """
    presets = dict(BUILTIN_PRESETS)
    for entry in data.get("presets", []):
        if "name" not in entry:
            raise SystemExit("each [[presets]] entry needs a 'name'")
        name = entry["name"]
        base = presets.get(name)

        fmt = entry.get("format", base.format if base else "openai")
        if fmt not in ("anthropic", "openai"):
            raise SystemExit(f"presets.{name}.format must be 'anthropic' or 'openai', got {fmt!r}")

        api_key_env = entry.get("api_key_env", list(base.api_key_env) if base else [])
        if isinstance(api_key_env, str):
            api_key_env = [api_key_env]

        if "base_url" not in entry and base is None:
            raise SystemExit(f"preset '{name}' needs a base_url")
        if "model" not in entry and base is None:
            raise SystemExit(f"preset '{name}' needs a model")

        presets[name] = Preset(
            name=name,
            format=fmt,
            base_url=entry.get("base_url", base.base_url if base else ""),
            model=entry.get("model", base.model if base else ""),
            api_key_env=tuple(api_key_env),
            key_file=entry.get("key_file", base.key_file if base else None),
            # Default thinking=False for brand-new (non-builtin) presets: most
            # OpenAI-compatible endpoints don't support DeepSeek-style thinking.
            thinking=bool(entry.get("thinking", base.thinking if base else False)),
            max_tokens=int(entry.get("max_tokens", base.max_tokens if base else 8192)),
            thinking_budget=int(entry.get("thinking_budget", base.thinking_budget if base else 4096)),
            temperature=entry.get("temperature", base.temperature if base else None),
            context_limit=int(entry.get("context_limit", base.context_limit if base else 128_000)),
            max_completion_tokens=bool(entry.get(
                "max_completion_tokens", base.max_completion_tokens if base else False)),
            extra_body=dict(entry.get("extra_body", base.extra_body if base else {})),
        )
    return presets


# ---------------------------------------------------------------------------
# Filesystem permissions (unchanged from before)
# ---------------------------------------------------------------------------

@dataclass
class PathRule:
    """A permission override for a path (directory or exact file).

    Each of read/write/delete may be None (inherit the global default) or one
    of allow/confirm/deny. A directory rule covers everything beneath it.
    """

    path: Path
    read: Permission | None = None
    write: Permission | None = None
    delete: Permission | None = None

    def specificity(self) -> int:
        return len(self.path.parts)


@dataclass
class Policy:
    """Filesystem permissions as a global default plus per-path overrides.

    For each operation (read/write/delete) the effective level is found by:
      1. Take the most specific matching PathRule that sets that operation.
      2. If none matches, use the global default.

    bash is intentionally NOT part of this path system -- a shell command isn't
    bound to a path and could escape any boundary, so it stays a single global
    level (default confirm) plus the allow/deny lists.
    """

    read: Permission = "allow"
    write: Permission = "confirm"
    delete: Permission = "confirm"

    rules: list[PathRule] = field(default_factory=list)

    bash: Permission = "confirm"
    bash_allowlist: list[str] = field(default_factory=list)
    bash_denylist: list[str] = field(
        default_factory=lambda: ["rm -rf /", "mkfs", ":(){:|:&};:", "dd if="]
    )

    read_only: bool = False

    def _matching_rules(self, target: Path) -> list[PathRule]:
        try:
            rt = target.resolve()
        except (OSError, RuntimeError):
            rt = target
        hits = []
        for r in self.rules:
            if rt == r.path:
                hits.append(r)
                continue
            try:
                rt.relative_to(r.path)
                hits.append(r)
            except ValueError:
                continue
        hits.sort(key=lambda r: r.specificity(), reverse=True)
        return hits

    def effective(self, op: str, target: Path) -> Permission:
        if self.read_only and op in ("write", "delete"):
            return "deny"
        for r in self._matching_rules(target):
            val = getattr(r, op)
            if val is not None:
                return val
        return getattr(self, op)

    def set_rule(
        self,
        path: Path,
        *,
        read: Permission | None = None,
        write: Permission | None = None,
        delete: Permission | None = None,
    ) -> None:
        path = path.resolve()
        for r in self.rules:
            if r.path == path:
                if read is not None:
                    r.read = read
                if write is not None:
                    r.write = write
                if delete is not None:
                    r.delete = delete
                return
        self.rules.append(PathRule(path=path, read=read, write=write, delete=delete))

    def clear_rule(self, path: Path) -> bool:
        path = path.resolve()
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.path != path]
        return len(self.rules) < before


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = "deepseek-v4-flash"
    format: str = "anthropic"  # "anthropic" | "openai" -- selects the wire client
    max_tokens: int = 8192
    thinking: bool = True
    thinking_budget: int = 4096
    temperature: float | None = None
    max_completion_tokens: bool = False
    project_root: Path = field(default_factory=Path.cwd)
    policy: Policy = field(default_factory=Policy)
    audit_log: Path | None = None
    markdown: bool = True
    thinking_display: str = "peek"

    # Transient-failure retries (429/5xx, connect and read errors).
    max_retries: int = 4
    # Context management. When the input tokens the API reports for a turn
    # exceed context_limit * compact_threshold, older history is summarized
    # away. Every token in history is re-sent (and re-billed) on every single
    # turn, so an un-managed session gets expensive well before it errors.
    context_limit: int = 128_000
    auto_compact: bool = True
    compact_threshold: float = 0.75
    compact_keep_last: int = 4  # recent messages kept verbatim when compacting

    preset_name: str = DEFAULT_PRESET
    presets: dict = field(default_factory=dict)  # name -> Preset, for switching
    extra_body: dict = field(default_factory=dict)
    global_key_file: str | None = None  # root-level key_file fallback

    def with_preset(self, name: str) -> "Config":
        """Switch to a different preset by name/alias, re-resolving its API key.

        If `name` doesn't match any known preset, falls back to the old
        behavior of just overriding the model string on the current
        format/base_url (handy for a one-off DeepSeek model id).
        """
        key = PRESET_ALIASES.get(name, name)
        preset = self.presets.get(key)
        if preset is None:
            return replace(self, model=name)
        api_key = _resolve_preset_api_key(preset, self.global_key_file)
        return replace(
            self,
            preset_name=preset.name,
            format=preset.format,
            base_url=preset.base_url,
            model=preset.model,
            api_key=api_key,
            thinking=preset.thinking,
            max_tokens=preset.max_tokens,
            thinking_budget=preset.thinking_budget,
            temperature=preset.temperature,
            context_limit=preset.context_limit,
            max_completion_tokens=preset.max_completion_tokens,
            extra_body=dict(preset.extra_body),
        )

    def with_model(self, name: str) -> "Config":
        """Backward-compat alias for with_preset (used by /model and -m)."""
        return self.with_preset(name)


def _read_toml(path: Path) -> dict:
    if not path.is_file() or tomllib is None:
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not parse {path}: {e}", file=sys.stderr)
        return {}


def _build_policy(data: dict, project_root: Path) -> Policy:
    p = Policy()
    pol = data.get("policy", {})

    def _perm(field_name: str, raw) -> str:
        v = normalize_permission(str(raw))
        if v not in ("allow", "confirm", "deny"):
            raise SystemExit(
                f"policy.{field_name} must be allow/confirm/deny (or forbid), got {raw!r}"
            )
        return v

    for fld in ("read", "write", "delete", "bash"):
        if fld in pol:
            setattr(p, fld, _perm(fld, pol[fld]))

    if "bash_allowlist" in pol:
        p.bash_allowlist = list(pol["bash_allowlist"])
    if "bash_denylist" in pol:
        p.bash_denylist = list(set(p.bash_denylist) | set(pol["bash_denylist"]))
    p.read_only = bool(pol.get("read_only", False))

    for entry in pol.get("rules", []):
        if "path" not in entry:
            raise SystemExit("each policy.rules entry needs a 'path'")
        raw_path = Path(os.path.expanduser(entry["path"]))
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path
        kwargs = {}
        for op in ("read", "write", "delete"):
            if op in entry:
                kwargs[op] = _perm(f"rules.{op}", entry[op])
        p.set_rule(raw_path.resolve(), **kwargs)

    return p


def load_config(
    *,
    project_root: Path | None = None,
    model: str | None = None,
    thinking: bool | None = None,
    read_only: bool | None = None,
) -> Config:
    root = (project_root or Path.cwd()).resolve()

    user_cfg = Path(os.path.expanduser("~/.config/lindwyrm/config.toml"))
    proj_cfg = root / ".lindwyrm.toml"

    data: dict = {}
    data.update(_read_toml(user_cfg))
    data.update(_read_toml(proj_cfg))  # project overrides user

    presets = _build_presets(data)
    policy = _build_policy(data, root)
    global_key_file = data.get("key_file")

    # Which preset to start on: default_preset (new), else legacy `model` key
    # (may be "flash"/"pro"/a preset name/a bare model id), else deepseek-flash.
    start_name = data.get("default_preset") or data.get("model") or DEFAULT_PRESET
    start_key = PRESET_ALIASES.get(start_name, start_name)
    preset = presets.get(start_key)
    if preset is None:
        # Legacy: `model` was a raw DeepSeek model id, not a preset name.
        preset = replace(BUILTIN_PRESETS[DEFAULT_PRESET], model=start_name)

    api_key = _resolve_preset_api_key(preset, global_key_file)

    cfg = Config(
        api_key=api_key,
        base_url=data.get("base_url", preset.base_url),  # legacy root override
        model=preset.model,
        format=preset.format,
        max_tokens=int(data.get("max_tokens", preset.max_tokens)),
        thinking=bool(data.get("thinking", preset.thinking)),
        thinking_budget=int(data.get("thinking_budget", preset.thinking_budget)),
        temperature=data.get("temperature", preset.temperature),
        max_completion_tokens=bool(data.get(
            "max_completion_tokens", preset.max_completion_tokens)),
        project_root=root,
        policy=policy,
        audit_log=Path(os.path.expanduser(data["audit_log"])) if data.get("audit_log") else None,
        markdown=bool(data.get("markdown", True)),
        thinking_display=str(data.get("thinking_display", "peek")),
        max_retries=max(1, int(data.get("max_retries", 4))),
        context_limit=int(data.get("context_limit", preset.context_limit)),
        auto_compact=bool(data.get("auto_compact", True)),
        compact_threshold=float(data.get("compact_threshold", 0.75)),
        compact_keep_last=max(0, int(data.get("compact_keep_last", 4))),
        preset_name=preset.name,
        presets=presets,
        extra_body=dict(preset.extra_body),
        global_key_file=global_key_file,
    )
    if cfg.thinking_display not in ("peek", "show", "hide"):
        cfg.thinking_display = "peek"

    # CLI overrides.
    if model is not None:
        cfg = cfg.with_preset(model)
    if thinking is not None:
        cfg.thinking = thinking
    if read_only is not None:
        cfg.policy.read_only = read_only

    cfg.max_tokens = min(cfg.max_tokens, MODEL_MAX_TOKENS_CEILING)
    return cfg
