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
# Proxies
# ---------------------------------------------------------------------------
#
# Three states have to be distinguishable, and TOML has no null:
#   key absent   -> inherit (preset falls back to the global setting)
#   "url"        -> use this proxy
#   "" / false   -> go direct, ignoring the global setting
#   "system"     -> honor HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
#
# Internally None means "inherit" and the empty string means "direct", so the
# resolved value is always a plain string. Environment proxies are NEVER used
# unless "system" is asked for explicitly: a proxy is an explicit choice here,
# and a stray ALL_PROXY silently rerouting an agent's API traffic is a nasty
# surprise.

PROXY_DIRECT = ""
PROXY_SYSTEM = "system"

# httpx supports exactly these; socks4 is not implemented by httpcore.
PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")


def parse_proxy(raw, where: str) -> str:
    """Validate one proxy setting and normalize it to its internal form."""
    if raw is False or raw is None:
        return PROXY_DIRECT
    if not isinstance(raw, str):
        raise SystemExit(f"{where}: proxy must be a string or false, got {raw!r}")

    value = raw.strip()
    if value == "" or value.lower() in ("direct", "none", "off"):
        return PROXY_DIRECT
    if value.lower() == PROXY_SYSTEM:
        return PROXY_SYSTEM

    scheme = value.split("://", 1)[0].lower() if "://" in value else ""
    if scheme not in PROXY_SCHEMES:
        extra = ""
        if scheme == "socks4":
            extra = " (socks4 is not supported by httpx; use socks5)"
        raise SystemExit(
            f"{where}: proxy must start with one of "
            f"{', '.join(s + '://' for s in PROXY_SCHEMES)}, "
            f"or be \"system\"/\"direct\"; got {value.split('://')[0]!r}{extra}"
        )
    if scheme.startswith("socks"):
        try:
            import socksio  # noqa: F401
        except ImportError:
            raise SystemExit(
                f"{where}: a SOCKS proxy is configured but the 'socksio' "
                f"package is missing. Install it with: pip install 'lindwyrm[socks]'"
            ) from None
    return value


def mask_proxy(proxy: str) -> str:
    """Proxy string safe to print: any password is replaced with ***.

    Display only -- the real value is what gets handed to httpx.
    """
    if not proxy or proxy == PROXY_SYSTEM or "@" not in proxy:
        return proxy or "direct"
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(proxy)
        if parts.password is None:
            return proxy
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        netloc = f"{parts.username or ''}:***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except ValueError:
        # Unparseable: better to show nothing than to risk showing a password.
        return f"{proxy.split('://', 1)[0]}://***"


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
    # None means "inherit the global setting"; see PROXY_DIRECT / PROXY_SYSTEM.
    proxy: str | None = None
    # Prices in currency units per MILLION tokens, the unit providers quote.
    # Left unset: no numbers are invented, and cost display stays off until
    # you fill in what your provider actually charges you.
    price_input: float | None = None
    price_output: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None
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


def _resolve_preset_api_key(preset: Preset, global_key_file: str | None,
                            required: bool = True) -> str:
    for env_name in preset.api_key_env:
        val = os.environ.get(env_name)
        if val:
            return val.strip()
    key_file = preset.key_file or global_key_file
    if key_file:
        p = Path(os.path.expanduser(key_file))
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    if not required:
        # Some commands only touch local files and have no business demanding
        # credentials -- /init writing a template shouldn't need an API key.
        return ""
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
            proxy=(parse_proxy(entry["proxy"], f"presets.{name}.proxy")
                   if "proxy" in entry else (base.proxy if base else None)),
            price_input=entry.get("price_input", base.price_input if base else None),
            price_output=entry.get("price_output", base.price_output if base else None),
            price_cache_read=entry.get("price_cache_read",
                                       base.price_cache_read if base else None),
            price_cache_write=entry.get("price_cache_write",
                                        base.price_cache_write if base else None),
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
    # Resolved proxy: "" = direct, "system" = use env vars, else a proxy URL.
    proxy: str = PROXY_DIRECT
    # Per-million-token prices for the active preset; None = don't show cost.
    price_input: float | None = None
    price_output: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None
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
    compact_keep_last: int = 4  # floor on messages kept verbatim
    # Protected zone, in tokens. A message count alone is a poor measure:
    # four messages can be forty tokens or half the window depending on
    # whether a test run landed in them.
    compact_keep_tokens: int = 8_000

    # Offloading: move bulky tool results to disk and leave a stub the model
    # can expand with read_offloaded. Reversible, unlike summarizing, so it
    # runs first when context gets tight.
    offload: bool = True
    offload_threshold_tokens: int = 1_000   # worth moving once out of the zone
    offload_eager_tokens: int = 8_000       # so big it goes the moment it appears

    # Path to the project instructions file. Unset means look for
    # LINDWYRM.md then AGENTS.md in the project root.
    context_file: str | None = None

    # Write each turn to ~/.local/share/lindwyrm/sessions so a closed
    # terminal doesn't lose the conversation.
    save_sessions: bool = True
    session_retention_days: int = 30

    preset_name: str = DEFAULT_PRESET
    presets: dict = field(default_factory=dict)  # name -> Preset, for switching
    extra_body: dict = field(default_factory=dict)
    global_key_file: str | None = None  # root-level key_file fallback
    global_proxy: str = PROXY_DIRECT    # root-level proxy, for presets that inherit
    # Set by --proxy: outranks both the preset and the global setting, and
    # survives /model, which a plain global default would not.
    proxy_override: str | None = None
    # Extra hosts that always go direct. Loopback is exempt unconditionally
    # and is not listed here; this is for a model server on the LAN.
    no_proxy: tuple[str, ...] = ()

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
            proxy=self.resolve_proxy(preset),
            price_input=preset.price_input,
            price_output=preset.price_output,
            price_cache_read=preset.price_cache_read,
            price_cache_write=preset.price_cache_write,
            extra_body=dict(preset.extra_body),
        )

    def resolve_proxy(self, preset: Preset) -> str:
        """Effective proxy for `preset`: --proxy, else the preset's own
        setting, else the global default."""
        if self.proxy_override is not None:
            return self.proxy_override
        if preset.proxy is not None:
            return preset.proxy
        return self.global_proxy

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
    require_key: bool = True,
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
    global_proxy = parse_proxy(data["proxy"], "proxy") if "proxy" in data else PROXY_DIRECT
    raw_no_proxy = data.get("no_proxy", [])
    if isinstance(raw_no_proxy, str):
        raw_no_proxy = [raw_no_proxy]
    no_proxy = tuple(str(h).strip() for h in raw_no_proxy if str(h).strip())

    # Which preset to start on: default_preset (new), else legacy `model` key
    # (may be "flash"/"pro"/a preset name/a bare model id), else deepseek-flash.
    start_name = data.get("default_preset") or data.get("model") or DEFAULT_PRESET
    start_key = PRESET_ALIASES.get(start_name, start_name)
    preset = presets.get(start_key)
    if preset is None:
        # Legacy: `model` was a raw DeepSeek model id, not a preset name.
        preset = replace(BUILTIN_PRESETS[DEFAULT_PRESET], model=start_name)

    api_key = _resolve_preset_api_key(preset, global_key_file, require_key)

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
        proxy=preset.proxy if preset.proxy is not None else global_proxy,
        global_proxy=global_proxy,
        price_input=preset.price_input,
        price_output=preset.price_output,
        price_cache_read=preset.price_cache_read,
        price_cache_write=preset.price_cache_write,
        no_proxy=no_proxy,
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
        compact_keep_tokens=max(0, int(data.get("compact_keep_tokens", 8_000))),
        offload=bool(data.get("offload", True)),
        offload_threshold_tokens=max(1, int(data.get("offload_threshold_tokens", 1_000))),
        offload_eager_tokens=max(1, int(data.get("offload_eager_tokens", 8_000))),
        context_file=data.get("context_file"),
        save_sessions=bool(data.get("save_sessions", True)),
        session_retention_days=max(1, int(data.get("session_retention_days", 30))),
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
