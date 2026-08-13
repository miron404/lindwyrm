"""Streaming terminal output.

The constraint that shapes all of this: `rich.Live` paints a fixed region of
the screen and cannot scroll. Feed it a renderable taller than the terminal
and Rich crops it, leaving an ellipsis at the bottom -- so a long answer or a
long stretch of reasoning appears to freeze, showing its first screenful
while the real stream races ahead unseen. It only ever "unfreezes" when the
region closes for a tool line or a confirmation prompt.

So nothing long-lived goes inside Live. The live region shows only a bounded
TAIL of whatever is arriving, hard-wrapped so it can never outgrow the
terminal. Completed pieces are printed to the console normally, which scrolls
and lands in scrollback where it can be read back.

Thinking display modes:
  "peek" - tail streams live, then disappears when the answer starts
  "show" - tail streams live, and the whole thing is kept in scrollback
  "hide" - not displayed, but a spinner shows the model is working
"""

from __future__ import annotations

import textwrap
import time

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.spinner import Spinner
    from rich.text import Text
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

# Rows of live tail to show. Small enough to fit any terminal, large enough
# to read as movement rather than flicker.
TAIL_LINES = 8


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
        self._segment_think = ""  # thinking since the last flush
        self._live: "Live | None" = None
        self._live_kind: str | None = None  # "thinking" | "text" | "waiting"
        self._started = 0.0
        self.last_thinking = ""  # retained for /think after the turn

    # -- lifecycle ---------------------------------------------------------

    def begin_turn(self) -> None:
        self._text_buf = ""
        self._think_buf = ""
        self._segment_think = ""
        self._live = None
        self._live_kind = None
        self._started = time.monotonic()
        self.wait()

    def end_turn(self) -> None:
        self._flush_text()
        self._flush_thinking()
        self._close_live()
        self.last_thinking = self._think_buf
        if self.enabled and self.console is not None:
            self.console.print()  # spacing after the answer

    def wait(self) -> None:
        """Show that the model is working before any output arrives.

        Without this, `hide` mode and any slow first token look identical to
        a hang -- there is simply nothing on screen.
        """
        if not self.enabled or self.console is None:
            return
        if self._live_kind in ("text", "thinking"):
            return
        self._open_live("waiting")
        self._live.update(self._status("working"))

    # -- internal ----------------------------------------------------------

    def _elapsed(self) -> str:
        seconds = time.monotonic() - self._started
        return f"{seconds:4.0f}s"

    def _status(self, label: str) -> "Text":
        return Text(f"  {label}… {self._elapsed()}", style="dim")

    def _spinner(self, label: str):
        return Spinner("dots", text=Text(f" {label}… {self._elapsed()}",
                                         style="dim"), style="dim")

    def _tail(self, text: str, lines: int = TAIL_LINES) -> str:
        """Last `lines` DISPLAY rows of `text`, hard-wrapped to the terminal.

        Wrapping is done here rather than left to Rich because a single very
        long logical line wraps to many rows, and it is rows -- not lines --
        that have to stay under the terminal height.
        """
        width = max(20, (self.console.width if self.console else 80) - 4)
        rows: list[str] = []
        for logical in text.splitlines():
            if logical.strip():
                rows.extend(textwrap.wrap(logical, width) or [""])
            else:
                rows.append("")
        return "\n".join(rows[-lines:])

    def _close_live(self) -> None:
        if self._live is not None:
            self._live.stop()  # transient=True, so the region is cleared
            self._live = None
            self._live_kind = None

    def _open_live(self, kind: str) -> None:
        if self._live_kind == kind:
            return
        self._close_live()
        # Always transient: the live region is scratch space. Anything worth
        # keeping is printed to the console separately.
        self._live = Live(console=self.console, transient=True,
                          refresh_per_second=10, auto_refresh=True)
        self._live.start()
        self._live_kind = kind

    def _flush_text(self) -> None:
        """Print the finished answer segment into scrollback, as Markdown."""
        if not self._text_buf:
            return
        if self.enabled and self.console is not None:
            self._close_live()
            self.console.print(Markdown(self._text_buf))
        self._text_buf = ""

    def _flush_thinking(self) -> None:
        """In `show` mode, keep the reasoning segment in scrollback."""
        segment = self._segment_think
        self._segment_think = ""
        if not segment.strip() or self.thinking_mode != "show":
            return
        if self.enabled and self.console is not None:
            self._close_live()
            self.console.print(Text("thinking", style="dim bold"))
            self.console.print(Text(segment.strip(), style="dim italic"))
            self.console.print()

    # -- streaming callbacks ----------------------------------------------

    def on_thinking(self, chunk: str) -> None:
        self._think_buf += chunk
        self._segment_think += chunk
        if not self.enabled:
            if self.thinking_mode != "hide":
                _plain_thinking(chunk)
            return

        if self.thinking_mode == "hide":
            # Content stays hidden, but the spinner proves it isn't wedged.
            self._open_live("waiting")
            self._live.update(self._spinner("thinking"))
            return

        self._open_live("thinking")
        self._live.update(Group(
            Text(f"thinking… {self._elapsed()}", style="dim bold"),
            Text(self._tail(self._segment_think), style="dim italic"),
        ))

    def on_text(self, chunk: str) -> None:
        if not self.enabled:
            return _plain_text(chunk)

        if self._live_kind != "text":
            # The answer has started: reasoning is done with for this segment.
            self._flush_thinking()
            self._open_live("text")
        self._text_buf += chunk
        self._live.update(Text(self._tail(self._text_buf)))

    def on_tool(self, name: str, tool_input: dict) -> None:
        # Everything so far is final; print it before the tool line so the
        # transcript reads in order.
        self._flush_text()
        self._flush_thinking()
        self._close_live()
        arg = ""
        for key in ("path", "command", "pattern", "ref"):
            if key in tool_input:
                arg = str(tool_input[key])
                break
        if self.enabled and self.console is not None:
            self.console.print(f"[cyan]→ {name}[/cyan] [dim]{_esc(arg)}[/dim]")
        else:
            print(f"\n→ {name} {arg}")

    def after_tool_result(self) -> None:
        """Called once a tool has reported: the next model call is starting."""
        self.wait()

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


def _esc(s: str) -> str:
    return s.replace("[", "\\[")


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
