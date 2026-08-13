"""lindwyrm command-line interface.

Usage:
    lwyrm                      # interactive REPL in the current directory
    lwyrm -m pro               # use the deepseek-pro preset
    lwyrm -m kimi              # use a custom preset you defined in config
    lwyrm --no-thinking        # disable thinking mode
    lwyrm --read-only          # never write/run, just answer
    lwyrm -p "fix the bug"     # one-shot: run a single prompt and exit

Presets bundle a provider + model + API key source. DeepSeek Flash/Pro
("flash"/"pro") ship built in; add your own via [[presets]] in
~/.config/lindwyrm/config.toml or ./.lindwyrm.toml to point at any other
Anthropic- or OpenAI-compatible endpoint. See lindwyrm.example.toml.

Slash commands inside the REPL:
    /model <name>               switch preset (flash|pro|<your preset name>)
    /presets                    list available presets
    /thinking <on|off>          toggle thinking mode (whether the model reasons)
    /think [peek|show|hide]     how thinking is displayed; no arg = reprint last
    /markdown <on|off>          toggle rich markdown rendering
    /perm <path> read=.. write=.. delete=..   set per-path permissions
                                (levels: allow|confirm|deny; "reset" clears;
                                 no args shows the table; path alone shows it)
    /policy                     show current permissions
    /sessions                   list saved sessions for this project
    /init                       write a starter AGENTS.md for this project
    /proxy                      show the proxy in use for this preset
    /context                    show how full the context window is
    /compact [instructions]     compact now; optional focus, e.g.
                                /compact keep the API decisions, drop debugging
    /clear                      clear conversation history
    /help                       show this help
    /exit, /quit                leave
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .config import Config, load_config, mask_proxy, parse_proxy
from .http import APIError, close_client
from . import offload
from .offload import get_store
from .project import CONTEXT_FILENAMES, TEMPLATE, find_context_file
from .render import Renderer, _HAS_RICH
from .sandbox import UserQuit, reset_session_grants
from .session import (
    describe_age,
    latest_session_id,
    list_sessions,
    load_session,
    new_session_id,
    save_session,
    sweep_sessions,
)

try:
    from rich.console import Console as _RichConsole
    _console = _RichConsole()
except ImportError:  # pragma: no cover
    _console = None

# Enables arrow-key cursor movement, line editing, and up/down history for the
# built-in input() prompts. Pure stdlib; a no-op on the rare platform without it.
try:
    import readline  # noqa: F401
except ImportError:  # pragma: no cover
    readline = None

# ANSI colors
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# How many lines of bash command output to echo to the terminal. The model
# always receives the full output; this only limits what you see.
BASH_OUTPUT_LINES = 25


def _banner(cfg: Config, agent=None) -> None:
    print(f"{BOLD}lindwyrm{RESET} {DIM}- coding agent{RESET}")
    print(f"  preset:   {CYAN}{cfg.preset_name}{RESET} {DIM}({cfg.format}, {cfg.model}){RESET}")
    print(f"  thinking: {'on' if cfg.thinking else 'off'}")
    print(f"  root:     {cfg.project_root}")
    p = cfg.policy
    print(f"  perms:    read={p.read} write={p.write} delete={p.delete} bash={p.bash}")
    if p.rules:
        print(f"  {DIM}{len(p.rules)} path rule(s) — see /policy{RESET}")
    if cfg.proxy:
        print(f"  proxy:    {CYAN}{mask_proxy(cfg.proxy)}{RESET}")
    if agent is not None and agent.context_files:
        # Worth surfacing: these files steer the agent, and on a cloned repo
        # you did not write them.
        print(f"  context:  {CYAN}{', '.join(agent.context_files)}{RESET}")
    if cfg.policy.read_only:
        print(f"  {YELLOW}read-only mode{RESET}")
    print(f"  {DIM}/help for commands, /exit to quit{RESET}\n")


def _print_policy(cfg: Config) -> None:
    p = cfg.policy
    print(f"{BOLD}Permissions{RESET}")
    print(f"  defaults: read={p.read} write={p.write} delete={p.delete}")
    print(f"  bash:     {p.bash}  (not path-scoped)")
    print(f"  read-only mode: {p.read_only}")
    if p.bash_allowlist:
        print(f"  bash allowlist: {', '.join(p.bash_allowlist)}")
    print(f"  bash denylist:  {', '.join(p.bash_denylist)}")
    if p.rules:
        print(f"  {BOLD}path rules{RESET} (most specific wins):")
        for r in sorted(p.rules, key=lambda r: r.specificity(), reverse=True):
            parts = []
            for op in ("read", "write", "delete"):
                v = getattr(r, op)
                if v is not None:
                    parts.append(f"{op}={v}")
            disp = _rel_display(cfg, r.path)
            print(f"    {disp}: {' '.join(parts) if parts else '(no overrides)'}")
    else:
        print("  (no path rules)")


def _print_sessions(cfg: Config) -> None:
    sessions = list_sessions(cfg.project_root)
    if not sessions:
        print(f"  {DIM}no saved sessions for this project yet{RESET}")
        return
    print(f"{BOLD}Sessions{RESET} {DIM}(this project){RESET}")
    for entry in sessions:
        print(f"  {CYAN}{entry['id']}{RESET}  {describe_age(entry['updated']):>9}  "
              f"{entry['messages']:>3} msg  {DIM}{entry['title']}{RESET}")
    print(f"  {DIM}resume with: lwyrm --resume <id>{RESET}")


def _init_context_file(cfg: Config) -> None:
    """Write a starter project instructions file, never over an existing one."""
    existing = find_context_file(cfg.project_root)
    if existing:
        print(f"  {YELLOW}{existing.name} already exists{RESET} "
              f"{DIM}({existing}){RESET}")
        print(f"  {DIM}edit it directly; /init won't overwrite your notes{RESET}")
        return
    target = cfg.project_root / CONTEXT_FILENAMES[-1]  # AGENTS.md
    try:
        target.write_text(TEMPLATE, encoding="utf-8")
    except OSError as e:
        print(f"  {RED}could not write {target}:{RESET} {e}")
        return
    print(f"  {GREEN}created {target.name}{RESET} {DIM}({target}){RESET}")
    print(f"  {DIM}fill it in, then restart the session to load it{RESET}")


def _print_proxy(cfg: Config) -> None:
    """Show the effective proxy. Any password is masked -- never printed."""
    print(f"{BOLD}Proxy{RESET}")
    if cfg.proxy == "system":
        print(f"  {CYAN}system{RESET} {DIM}(HTTP_PROXY / HTTPS_PROXY / ALL_PROXY){RESET}")
    elif cfg.proxy:
        print(f"  {CYAN}{mask_proxy(cfg.proxy)}{RESET} {DIM}(preset: {cfg.preset_name}){RESET}")
    else:
        print(f"  direct {DIM}(no proxy; environment variables are ignored){RESET}")
    if cfg.global_proxy and cfg.global_proxy != cfg.proxy:
        print(f"  {DIM}global default: {mask_proxy(cfg.global_proxy)} "
              f"— overridden by this preset{RESET}")


def _print_context(cfg: Config, agent) -> None:
    used = agent.context_tokens()
    limit = cfg.context_limit
    pct = agent.context_fraction() * 100
    measured = "measured" if agent.last_input_tokens else "estimated"
    bar_width = 24
    filled = min(bar_width, int(bar_width * pct / 100))
    bar = "█" * filled + "·" * (bar_width - filled)
    color = RED if pct >= 90 else (YELLOW if pct >= cfg.compact_threshold * 100 else GREEN)
    print(f"{BOLD}Context{RESET}")
    print(f"  {color}{bar}{RESET} {pct:.0f}%  ({used:,} / {limit:,} tokens, {measured})")
    print(f"  messages: {len(agent.messages)}   output so far: {agent.total_output_tokens:,} tokens")
    if agent.total_cache_read_tokens:
        # Worth surfacing: a sudden drop means something upstream of the
        # history changed and broke the cacheable prefix.
        billed = agent.total_fresh_input_tokens + agent.total_cache_read_tokens
        share = agent.total_cache_read_tokens / max(1, billed) * 100
        print(f"  {DIM}served from cache: {agent.total_cache_read_tokens:,} input "
              f"tokens ({share:.0f}% of input){RESET}")
    _print_cost(cfg, agent)
    if agent.context_files:
        print(f"  {DIM}project context: {', '.join(agent.context_files)}{RESET}")
    store = get_store()
    if len(store):
        print(f"  {DIM}offloaded: {len(store)} result(s), "
              f"{store.total_chars() / 1024:.0f} KB on disk{RESET}")
    if cfg.auto_compact:
        print(f"  {DIM}auto-compacts at {cfg.compact_threshold * 100:.0f}%{RESET}")
    else:
        print(f"  {DIM}auto-compaction off — use /compact{RESET}")


def _print_cost(cfg: Config, agent) -> None:
    """Session spend, when the active preset has prices configured."""
    cost = agent.session_cost()
    if cost is None:
        return
    print(f"  {BOLD}spend{RESET}   {_money(cost)} {DIM}this session{RESET}")
    print(f"  {DIM}  fresh in {agent.total_fresh_input_tokens:,} · "
          f"cached in {agent.total_cache_read_tokens:,} · "
          f"cache writes {agent.total_cache_write_tokens:,} · "
          f"out {agent.total_output_tokens:,}{RESET}")


def _money(value: float) -> str:
    """Small amounts need more decimals than a currency format allows."""
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _turn_summary(cfg: Config, agent, before: float | None) -> None:
    """One dim line after each turn -- only once prices are configured, since
    setting them is the signal that you want to watch the meter."""
    cost = agent.session_cost()
    if cost is None:
        return
    delta = cost - (before or 0.0)
    print(f"{DIM}  {_money(delta)} this turn · {_money(cost)} session · "
          f"{agent.context_fraction() * 100:.0f}% context{RESET}")


def _make_retry_printer():
    """Callback that tells the user why the client is pausing before a retry."""
    def on_retry(attempt: int, reason: str, delay: float) -> None:
        print(f"  {YELLOW}{reason}{RESET} {DIM}— retrying in {delay:.1f}s "
              f"(attempt {attempt}){RESET}")
    return on_retry


def _rel_display(cfg: Config, p) -> str:
    try:
        from pathlib import Path
        return str(Path(p).relative_to(cfg.project_root))
    except (ValueError, TypeError):
        return str(p)


# --- live output callbacks ---

def _emit(text: str = "") -> None:
    """Print through the rich console if available, else plain."""
    if _console is not None:
        _console.print(text)
    else:
        print(text)


def make_tool_result_printer(renderer: Renderer):
    """Build an on_tool_result callback bound to the active renderer's console."""
    out = renderer.console if (renderer.enabled and renderer.console) else None

    def on_tool_result(name: str, result: str, is_error: bool) -> None:
        _print_tool_result(name, result, is_error)
        # Only now: the next model call is starting, and a spinner opened
        # before this printed would have been painted over.
        renderer.after_tool_result()

    def _print_tool_result(name: str, result: str, is_error: bool) -> None:
        tag = "error" if is_error else "ok"
        color = "red" if is_error else "green"
        lines = result.splitlines() if result else []

        if name == "bash" and lines:
            head = lines[0]
            body = lines[1:]
            if out is not None:
                out.print(f"  [{color}]{tag}[/{color}] [dim]{head}[/dim]")
                for ln in body[:BASH_OUTPUT_LINES]:
                    out.print(f"  [dim]│[/dim] {_esc(ln)}")
                hidden = len(body) - BASH_OUTPUT_LINES
                if hidden > 0:
                    out.print(f"  [dim]… {hidden} more line(s) hidden (full output sent to the model)[/dim]")
            else:
                print(f"  {tag} {head}")
                for ln in body[:BASH_OUTPUT_LINES]:
                    print(f"  │ {ln}")
                hidden = len(body) - BASH_OUTPUT_LINES
                if hidden > 0:
                    print(f"  … {hidden} more line(s) hidden (full output sent to the model)")
            return

        first = lines[0] if lines else ""
        if out is not None:
            out.print(f"  [{color}]{tag}[/{color}] [dim]{_esc(first[:120])}[/dim]")
        else:
            print(f"  {tag} {first[:120]}")

    return on_tool_result


def _esc(s: str) -> str:
    """Escape rich markup so file contents can't be interpreted as tags."""
    return s.replace("[", "\\[")


def _setup_history() -> None:
    """Load and persist REPL prompt history across sessions, and configure
    readline for correct line wrapping / multi-line paste."""
    if readline is None:
        return
    import os

    try:
        # Bracketed paste mode (GNU readline 8.1+): the terminal marks pasted
        # text so readline treats embedded newlines as literal characters
        # instead of "Enter pressed" -- without this, pasting a multi-line
        # message submits each line as a separate prompt.
        readline.parse_and_bind("set enable-bracketed-paste on")
        # Ctrl+J inserts a literal newline for composing a multi-line message
        # by hand; Enter still submits.
        readline.parse_and_bind(r'"\C-j": "\n"')
    except Exception:  # noqa: BLE001 -- best-effort; older/non-GNU readline may reject this
        pass

    hist_dir = Path(os.path.expanduser("~/.local/share/lindwyrm"))
    hist_file = hist_dir / "history"
    try:
        hist_dir.mkdir(parents=True, exist_ok=True)
        if hist_file.is_file():
            readline.read_history_file(str(hist_file))
        readline.set_history_length(1000)
        import atexit
        atexit.register(lambda: _save_history(str(hist_file)))
    except OSError:
        pass


def _save_history(path: str) -> None:
    if readline is None:
        return
    try:
        readline.write_history_file(path)
    except OSError:
        pass


# The prompt shown before each user turn. ANSI color codes are wrapped in
# \001...\002 (RL_PROMPT_START/END_IGNORE) so readline doesn't count them as
# visible characters when computing cursor position -- without this, typing
# or pasting text long enough to wrap onto a second terminal line causes the
# cursor and redraw to land in the wrong place. Those markers are only
# meaningful to readline itself; when stdin isn't a real terminal (piped
# input, tests) input() skips readline entirely and would print the raw
# marker bytes, so we only add them in interactive sessions.
def _make_prompt() -> str:
    if sys.stdin.isatty():
        return f"\001{BOLD}\002you ›\001{RESET}\002 "
    return f"{BOLD}you ›{RESET} "


PROMPT = _make_prompt()


def _make_agent(cfg: Config, args) -> Agent:
    """Build the agent, resuming a saved session when asked."""
    agent = Agent(cfg)
    if not cfg.save_sessions:
        return agent

    sweep_sessions(cfg.session_retention_days)
    wanted = args.resume if getattr(args, "resume", None) else None
    if wanted is None and getattr(args, "continue_", False):
        wanted = latest_session_id(cfg.project_root)
        if wanted is None:
            print(f"  {DIM}no previous session for this project; starting fresh{RESET}")

    if wanted:
        state = load_session(wanted)
        if state is None:
            print(f"  {RED}no such session:{RESET} {wanted}")
        else:
            agent.session_id = wanted
            agent.session_created = state.get("created")
            # The offload store has to exist under the session's own name
            # before restore(), or the recovered refs point at a directory
            # this process will never write to.
            offload.set_store(offload.OffloadStore(session_id=wanted))
            agent.restore(state)
            print(f"  {GREEN}resumed{RESET} {CYAN}{wanted}{RESET} "
                  f"{DIM}({len(agent.messages)} messages, "
                  f"{describe_age(state.get('updated', 0))}){RESET}")
            if state.get("preset") and state["preset"] != cfg.preset_name:
                print(f"  {YELLOW}note:{RESET} saved with preset "
                      f"{state['preset']}, now running {cfg.preset_name}")
            return agent

    agent.session_id = new_session_id()
    offload.set_store(offload.OffloadStore(session_id=agent.session_id))
    return agent


def run_repl(cfg: Config, args=None) -> None:
    _setup_history()
    agent = _make_agent(cfg, args) if args is not None else Agent(cfg)
    _banner(cfg, agent)
    renderer = Renderer(
        thinking_mode=cfg.thinking_display,
        enabled=cfg.markdown,
    )
    interrupted_once = False
    while True:
        try:
            line = input(PROMPT).strip()
            interrupted_once = False
        except EOFError:
            # Ctrl+D is the deliberate way out.
            print("\nbye")
            return
        except KeyboardInterrupt:
            # Ctrl+C abandons the line being typed, the way a shell does --
            # it is also what stops a running turn, and quitting the whole
            # session on it is far too easy to do by accident.
            if interrupted_once:
                print(f"\n{DIM}bye{RESET}")
                return
            interrupted_once = True
            print(f"\n{DIM}(^C again to quit, or Ctrl+D / /exit){RESET}")
            continue
        if not line:
            continue
        if line.startswith("/"):
            if _handle_command(line, cfg, agent, renderer):
                return
            continue
        agent.add_user(line)
        _do_turn(cfg, agent, renderer)


def _persist(cfg: Config, agent: Agent) -> None:
    """Write the session out. Called after every turn, not just at exit --
    the whole point is surviving a crash or a closed terminal."""
    if not cfg.save_sessions or not agent.session_id:
        return
    state = agent.snapshot()
    state["project_root"] = str(cfg.project_root)
    state["preset"] = cfg.preset_name
    state["model"] = cfg.model
    state["created"] = agent.session_created
    save_session(agent.session_id, state)


def _do_turn(cfg: Config, agent: Agent, renderer: Renderer) -> None:
    reset_session_grants()  # re-confirm "always" grants each user turn
    renderer.thinking_mode = cfg.thinking_display  # pick up /think changes
    renderer.enabled = cfg.markdown and _HAS_RICH
    renderer.begin_turn()
    on_tool_result = make_tool_result_printer(renderer)
    cost_before = agent.session_cost()
    try:
        _emit(f"[dim]{cfg.model}[/dim]" if _console else cfg.model)
        agent.run_turn(
            on_text=renderer.on_text,
            on_thinking=renderer.on_thinking if cfg.thinking else None,
            on_tool=renderer.on_tool,
            on_tool_result=on_tool_result,
            on_retry=_make_retry_printer(),
            on_notice=lambda msg: print(f"  {DIM}{msg}{RESET}"),
        )
        renderer.end_turn()
        _turn_summary(cfg, agent, cost_before)
        _persist(cfg, agent)
    except UserQuit:
        # The user chose [q]uit at a confirmation prompt: stop the turn, but
        # stay in the REPL rather than tearing the session down.
        renderer.end_turn()
        print(f"\n{YELLOW}(stopped at your request){RESET}\n")
    except KeyboardInterrupt:
        renderer.end_turn()
        print(f"\n{YELLOW}(interrupted){RESET}\n")
    except APIError as e:
        renderer.end_turn()
        print(f"\n{RED}API error:{RESET} {e}\n")
    except Exception as e:  # noqa: BLE001
        renderer.end_turn()
        print(f"\n{RED}error:{RESET} {type(e).__name__}: {e}\n")


def _handle_command(line: str, cfg: Config, agent: Agent, renderer: Renderer) -> bool:
    """Return True to exit the REPL."""
    parts = line.split()
    cmd = parts[0]
    if cmd in ("/exit", "/quit"):
        print("bye")
        return True
    if cmd == "/help":
        print(__doc__)
    elif cmd == "/model":
        if len(parts) < 2:
            print(f"current: {cfg.preset_name} ({cfg.format}, {cfg.model})")
        else:
            try:
                _switch_preset(cfg, parts[1], agent)
                print(f"-> {cfg.preset_name} ({cfg.format}, {cfg.model}, "
                      f"context {cfg.context_limit:,})")
            except SystemExit as e:
                print(f"  {RED}error:{RESET} {e}")
    elif cmd == "/presets":
        _list_presets(cfg)
    elif cmd == "/thinking":
        if len(parts) >= 2:
            cfg.thinking = parts[1].lower() in ("on", "true", "1", "yes")
        print(f"thinking: {'on' if cfg.thinking else 'off'}")
    elif cmd == "/think":
        # No arg: reprint last turn's thinking. With arg: set display mode.
        if len(parts) >= 2:
            mode = parts[1].lower()
            if mode in ("peek", "show", "hide"):
                cfg.thinking_display = mode
                renderer.thinking_mode = mode
                print(f"thinking display -> {mode}")
            else:
                print("usage: /think [peek|show|hide]  (no arg = show last thinking)")
        else:
            renderer.reprint_thinking()
    elif cmd == "/markdown":
        if len(parts) >= 2:
            cfg.markdown = parts[1].lower() in ("on", "true", "1", "yes")
            renderer.enabled = cfg.markdown and _HAS_RICH
        status = "on" if cfg.markdown else "off"
        if not _HAS_RICH:
            status += " (rich not installed -- plain output)"
        print(f"markdown: {status}")
    elif cmd == "/perm":
        _perm_command(cfg, parts[1:])
    elif cmd == "/policy":
        _print_policy(cfg)
    elif cmd == "/sessions":
        _print_sessions(cfg)
    elif cmd == "/init":
        _init_context_file(cfg)
    elif cmd == "/proxy":
        _print_proxy(cfg)
    elif cmd == "/context":
        _print_context(cfg, agent)
    elif cmd == "/compact":
        focus = line.split(None, 1)[1].strip() if len(parts) > 1 else ""
        moved, freed = agent.microcompact()
        if moved:
            print(f"  {GREEN}offloaded {moved} large tool result(s), "
                  f"~{freed} tokens freed{RESET}")
        print(f"  {DIM}summarizing older history…{RESET}")
        ok, note = agent.compact(focus=focus, on_retry=_make_retry_printer())
        print(f"  {GREEN if ok else YELLOW}{note}{RESET}")
        _print_context(cfg, agent)
    elif cmd == "/clear":
        agent.messages.clear()
        agent.last_input_tokens = 0
        # A fresh id: the previous session stays on disk rather than being
        # overwritten by the empty one that follows.
        if cfg.save_sessions:
            agent.session_id = new_session_id()
            agent.session_created = None
        print("history cleared (new session started)")
    else:
        print(f"unknown command: {cmd} (try /help)")
    return False


def _perm_command(cfg: Config, args: list[str]) -> None:
    """Handle /perm. Forms:
        /perm                         -> show all rules (same as /policy)
        /perm <path>                  -> show effective perms for path
        /perm <path> read=allow ...   -> set overrides for path
        /perm <path> reset            -> remove the path's rule
    """
    import os
    from .config import normalize_permission

    if not args:
        _print_policy(cfg)
        return

    raw = args[0]
    target = Path(os.path.expanduser(raw))
    if not target.is_absolute():
        target = cfg.project_root / target
    target = target.resolve()

    overrides = args[1:]

    if not overrides:
        # Show effective perms for this path.
        eff = {op: cfg.policy.effective(op, target) for op in ("read", "write", "delete")}
        print(f"  {_rel_display(cfg, target)}: "
              f"read={eff['read']} write={eff['write']} delete={eff['delete']}")
        return

    if overrides == ["reset"]:
        if cfg.policy.clear_rule(target):
            print(f"  rule removed: {_rel_display(cfg, target)}")
        else:
            print(f"  no rule for: {_rel_display(cfg, target)}")
        return

    # Parse op=level pairs.
    kwargs = {}
    for tok in overrides:
        if "=" not in tok:
            print(f"  {RED}bad token:{RESET} {tok!r} (expected op=level, e.g. write=confirm)")
            return
        op, _, level = tok.partition("=")
        op = op.strip().lower()
        if op not in ("read", "write", "delete"):
            print(f"  {RED}unknown op:{RESET} {op!r} (read|write|delete)")
            return
        lvl = normalize_permission(level)
        if lvl not in ("allow", "confirm", "deny"):
            print(f"  {RED}bad level:{RESET} {level!r} (allow|confirm|deny)")
            return
        kwargs[op] = lvl

    cfg.policy.set_rule(target, **kwargs)
    summary = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"  rule set (this session): {_rel_display(cfg, target)} -> {summary}")


# Every Config field a preset can carry. Kept next to Preset itself in spirit:
# forgetting one here means /model silently keeps the old provider's value.
PRESET_FIELDS = (
    "preset_name", "format", "base_url", "model", "api_key",
    "thinking", "max_tokens", "thinking_budget", "temperature",
    "context_limit", "max_completion_tokens", "proxy", "extra_body",
    "price_input", "price_output", "price_cache_read", "price_cache_write",
)


def _switch_preset(cfg: Config, name: str, agent: Agent | None = None) -> None:
    """Switch cfg to a different preset IN PLACE (cfg is shared by reference
    across the REPL, so we copy every field a preset switch can touch rather
    than replacing the object)."""
    new = cfg.with_model(name)  # may raise SystemExit if the key is missing
    for f in PRESET_FIELDS:
        setattr(cfg, f, getattr(new, f))
    if agent is not None:
        # The recorded input size came from the previous provider's tokenizer
        # and was measured against its context window. Keeping it would judge
        # the new model's fullness with the old model's numbers.
        agent.last_input_tokens = 0


def _list_presets(cfg: Config) -> None:
    from .config import PRESET_ALIASES

    print(f"{BOLD}Presets{RESET}")
    for name, preset in sorted(cfg.presets.items()):
        marker = f" {GREEN}(active){RESET}" if name == cfg.preset_name else ""
        aliases = [a for a, target in PRESET_ALIASES.items() if target == name]
        alias_s = f" [{', '.join(aliases)}]" if aliases else ""
        print(f"  {name}{alias_s}: {preset.format}, {preset.model}{marker}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lwyrm", description="Minimal CLI coding agent for Anthropic- or OpenAI-compatible providers")
    p.add_argument("-m", "--model", help="preset name (flash | pro | your custom preset)")
    p.add_argument("--no-thinking", action="store_true", help="disable thinking mode")
    p.add_argument("--thinking", action="store_true", help="force thinking mode on")
    p.add_argument("--read-only", action="store_true", help="never write/run anything")
    p.add_argument("--no-markdown", action="store_true", help="disable rich markdown rendering")
    p.add_argument("-C", "--dir", help="project root (default: cwd)")
    p.add_argument("-p", "--prompt", help="one-shot prompt; run and exit")
    p.add_argument("-c", "--continue", dest="continue_", action="store_true",
                   help="resume the most recent session for this project")
    p.add_argument("--resume", metavar="ID",
                   help="resume a specific session (see /sessions)")
    p.add_argument("--no-save", action="store_true",
                   help="don't write this session to disk")
    p.add_argument("--proxy", metavar="URL",
                   help="proxy for API traffic: socks5h://host:port, http://host:port, "
                        "'system' to use HTTP_PROXY/ALL_PROXY, or 'direct' to force none")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    thinking = None
    if args.no_thinking:
        thinking = False
    elif args.thinking:
        thinking = True

    try:
        cfg = load_config(
            project_root=Path(args.dir) if args.dir else None,
            model=args.model,
            thinking=thinking,
            read_only=True if args.read_only else None,
        )
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    if args.no_markdown:
        cfg.markdown = False

    if args.proxy is not None:
        # Recorded as an override rather than just a resolved value: it has
        # to outrank a preset's own proxy after /model too, which a plain
        # global default would not.
        try:
            cfg.proxy_override = cfg.proxy = parse_proxy(args.proxy, "--proxy")
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 2

    if args.no_save:
        cfg.save_sessions = False

    try:
        if args.prompt:
            agent = _make_agent(cfg, args)
            agent.add_user(args.prompt)
            renderer = Renderer(thinking_mode=cfg.thinking_display, enabled=cfg.markdown)
            _do_turn(cfg, agent, renderer)
            return 0

        run_repl(cfg, args)
        return 0
    finally:
        close_client()  # release the pooled connection


if __name__ == "__main__":
    raise SystemExit(main())
