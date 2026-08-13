"""Rich-based streaming output for lindwyrm.

Design goals, given how terminals work:
  * Stream the assistant's answer as Markdown, updating live so you watch it
    appear (code blocks, tables, bold, headings all render via rich).
  * Show thinking live but EPHEMERALLY: it streams in dim text inside a Live
    region, and the moment the real answer starts, the thinking region is
    cleared so the scrollback stays clean (only the final answer remains).
  * Degrade gracefully: if rich isn't installed, fall back to plain streaming.

A terminal is a line stream, not a widget tree, so there's no true "click to
expand". "Peek" thinking (transient=True Live that disappears) is the closest
clean equivalent. The full thinking text is still retained so a /think command
can reprint it on demand.

Thinking display modes:
  "peek" - stream live in a transient region, then clear when the answer starts
  "show" - stream live and KEEP it in scrollback (rendered as dim markdown)
  "hide" - don't display thinking at all
"""

from __future__ import annotations

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


class Renderer:
    """Stateful per-turn renderer. One instance per REPL session is fine;
    call begin_turn() before each turn."""

    def __init__(self, thinking_mode: str = "peek", enabled: bool = True) -> None:
        self.thinking_mode = thinking_mode  # peek | show | hide
        self.enabled = enabled and _HAS_RICH
        self.console = Console() if _HAS_RICH else None
        # Per-turn accumulators.
        self._text_buf = ""
        self._think_buf = ""
        self._live: "Live | None" = None
        self._live_kind: str | None = None  # "thinking" | "text"
        self.last_thinking = ""  # retained for /think after the turn

    # -- lifecycle ---------------------------------------------------------

    def begin_turn(self) -> None:
        self._text_buf = ""
        self._think_buf = ""
        self._live = None
        self._live_kind = None

    def end_turn(self) -> None:
        self._close_live(clear=False)
        self.last_thinking = self._think_buf
        if self.enabled and self.console is not None:
            self.console.print()  # spacing after the answer

    # -- internal Live management -----------------------------------------

    def _close_live(self, *, clear: bool) -> None:
        if self._live is not None:
            was_text = self._live_kind == "text"
            self._live.stop()
            # When transient=True, stopping clears the region automatically.
            self._live = None
            self._live_kind = None
            # A kept (non-transient) text block leaves the cursor at end of the
            # last line; add a blank line so the next output isn't glued on.
            if was_text and not clear and self.console is not None:
                self.console.print()

    def _open_live(self, kind: str, transient: bool) -> None:
        self._live = Live(
            console=self.console,
            refresh_per_second=12,
            transient=transient,
            auto_refresh=False,
        )
        self._live.start()
        self._live_kind = kind

    # -- streaming callbacks ----------------------------------------------

    def on_thinking(self, chunk: str) -> None:
        if self.thinking_mode == "hide":
            self._think_buf += chunk  # still retain for /think
            return
        if not self.enabled:
            return _plain_thinking(chunk)

        self._think_buf += chunk
        if self._live_kind != "thinking":
            self._close_live(clear=False)
            transient = self.thinking_mode == "peek"
            self._open_live("thinking", transient=transient)
        # Dim, plain (not markdown -- thinking is usually a raw stream).
        body = Text(self._think_buf, style="dim italic")
        header = Text("thinking\n", style="dim bold")
        self._live.update(Group(header, body), refresh=True)

    def on_text(self, chunk: str) -> None:
        if not self.enabled:
            return _plain_text(chunk)

        # First real text: drop the thinking region (peek) before starting.
        if self._live_kind != "text":
            self._close_live(clear=self.thinking_mode == "peek")
            self._open_live("text", transient=False)
        self._text_buf += chunk
        self._live.update(Markdown(self._text_buf), refresh=True)

    def on_tool(self, name: str, tool_input: dict) -> None:
        # Close any live region so the tool line prints cleanly below.
        self._close_live(clear=self.thinking_mode == "peek" and self._live_kind == "thinking")
        self._text_buf = ""  # a new text block may follow the tool result
        arg = ""
        for key in ("path", "command", "pattern"):
            if key in tool_input:
                arg = str(tool_input[key])
                break
        if self.enabled and self.console is not None:
            self.console.print(f"[cyan]→ {name}[/cyan] [dim]{arg}[/dim]")
        else:
            print(f"\n→ {name} {arg}")

    def reprint_thinking(self) -> None:
        """Used by /think: show the retained thinking of the last turn."""
        if not self.last_thinking.strip():
            msg = "(no thinking recorded for the last turn)"
            (self.console.print if self.console else print)(msg)
            return
        if self.enabled and self.console is not None:
            self.console.print(Text(self.last_thinking, style="dim italic"))
        else:
            print(self.last_thinking)


# -- plain fallbacks (no rich) --------------------------------------------

_plain_state = {"in_thinking": False}


def _plain_thinking(chunk: str) -> None:
    if not _plain_state["in_thinking"]:
        print("\n\033[2m\033[1m[thinking]\033[0m\033[2m", end="")
        _plain_state["in_thinking"] = True
    print(chunk, end="", flush=True)


def _plain_text(chunk: str) -> None:
    if _plain_state["in_thinking"]:
        print("\033[0m\n")
        _plain_state["in_thinking"] = False
    print(chunk, end="", flush=True)
