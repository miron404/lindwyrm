"""Streaming terminal output.

`rich.Live` paints a fixed region and cannot scroll: give it more than a
screenful and Rich crops it, and every repaint drags a scrolled-back terminal
back to the bottom. So Live is used as little as possible.

The answer never goes through it. Answer text only ever grows at the end, so
finished Markdown blocks are printed straight to the console: full width,
scrolls like any other program's output, and readable in scrollback. A block
ends at a blank line -- but never inside a fenced code block, which would
render half a fence -- with a length cap so one long paragraph doesn't sit
invisible while it is written.

Live is left for the two things that genuinely have to disappear afterwards:
the reasoning preview and the spinner. Both are bounded to a few rows, and
both repaint slowly, because repainting is what fights scrollback.

Thinking display modes:
  "peek" - a bounded tail streams live, then disappears when the answer starts
  "show" - the same tail, and the whole thing is kept in scrollback
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

# A block this long is released even without a blank line, so one very long
# paragraph doesn't sit invisible while it is being written.
MAX_PENDING_LINES = 12

# Live repaint rate, for the reasoning preview and the spinner. Every repaint
# drags a scrolled-back terminal to the bottom, so this trades scrollability
# for smoothness. It only applies while one of those two is on screen -- the
# answer itself is printed, not redrawn, so it is unaffected either way.
REFRESH_HZ = 10


class Renderer:
    """Stateful per-turn renderer. One instance per REPL session is fine;
    call begin_turn() before each turn."""

    def __init__(self, thinking_mode: str = "peek", enabled: bool = True) -> None:
        self.thinking_mode = thinking_mode  # peek | show | hide
        self.enabled = enabled and _HAS_RICH
        self.console = Console() if _HAS_RICH else None
        # Per-turn accumulators.
        self._pending = ""        # answer text not yet forming a whole block
        self._think_buf = ""
        self._segment_think = ""  # thinking since the last flush
        self._spinner = None      # created once; a fresh one never animates
        self._live: "Live | None" = None
        self._live_kind: str | None = None  # "thinking" | "text" | "waiting"
        self._started = 0.0
        self.last_thinking = ""  # retained for /think after the turn

    # -- lifecycle ---------------------------------------------------------

    def begin_turn(self) -> None:
        self._pending = ""
        self._think_buf = ""
        self._segment_think = ""
        self._live = None
        self._live_kind = None
        self._spinner = None
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
        self._live.update(self._get_spinner("working"))

    # -- internal ----------------------------------------------------------

    def _elapsed(self) -> str:
        seconds = time.monotonic() - self._started
        return f"{seconds:4.0f}s"

    def _get_spinner(self, label: str):
        """One Spinner for the whole wait.

        Building a new one per update restarts its frame counter, so it sits
        on frame zero forever and looks frozen -- which is worse than no
        spinner at all, since a frozen spinner reads as a hung process.
        """
        if self._spinner is None:
            self._spinner = Spinner("dots", style="dim")
        self._spinner.update(text=Text(f" {label}… {self._elapsed()}",
                                       style="dim"))
        return self._spinner

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
        # Every repaint yanks a scrolled-back terminal to the bottom, so this
        # is as slow as it can be while the spinner still reads as motion.
        self._live = Live(console=self.console, transient=True,
                          refresh_per_second=REFRESH_HZ, auto_refresh=True)
        self._live.start()
        self._live_kind = kind

    def _flush_text(self) -> None:
        """Print whatever answer text is still buffered."""
        if not self._pending.strip():
            self._pending = ""
            return
        if self.enabled and self.console is not None:
            self._print_block(self._pending)
        self._pending = ""

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
            self._live.update(self._get_spinner("thinking"))
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
            self._close_live()
            self._live_kind = "text"
        self._pending += chunk
        self._emit_ready_blocks()

    def _emit_ready_blocks(self) -> None:
        """Print every complete Markdown block that has arrived.

        The answer is append-only, so it does not belong in a Live region at
        all: printing finished blocks straight to the console lets the
        terminal scroll the way it does for any other program, at full width,
        instead of trapping the text in a small window that only opens up at
        the end.
        """
        while True:
            block, rest = self._take_block(self._pending)
            if block is None:
                return
            self._pending = rest
            self._print_block(block)

    def _take_block(self, buf: str) -> tuple[str | None, str]:
        """Split off one renderable Markdown block, if the buffer holds one.

        Blocks end at a blank line, but never inside a fenced code block --
        splitting there would render half a fence. A long block is released
        anyway so a single big paragraph doesn't sit invisible.
        """
        lines = buf.split("\n")
        if len(lines) < 2:
            return None, buf  # nothing but an unfinished line

        in_fence = False
        for i, line in enumerate(lines[:-1]):  # last piece is still partial
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if in_fence:
                    return "\n".join(lines[:i + 1]), "\n".join(lines[i + 1:])
                in_fence = True
                continue
            if in_fence:
                continue
            if not stripped and i > 0:
                return "\n".join(lines[:i]), "\n".join(lines[i + 1:])

        if not in_fence and len(lines) > MAX_PENDING_LINES:
            keep = lines[-1]
            return "\n".join(lines[:-1]), keep
        return None, buf

    def _print_block(self, block: str) -> None:
        if not block.strip():
            return
        if self.console is not None:
            self._close_live()
            self.console.print(Markdown(block))
            self._live_kind = "text"

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
