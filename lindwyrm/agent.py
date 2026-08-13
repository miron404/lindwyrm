"""The agentic loop: model -> tool calls -> tool results -> model -> ...

Keeps the full message history (including thinking blocks, which DeepSeek
requires echoed back). Stops when the model returns end_turn without
requesting more tools.

Because every token of history is re-sent on every turn, long sessions get
expensive and eventually hit the model's context limit. compact() summarizes
older history away; run_turn() calls it automatically once the API-reported
input size crosses a share of the context window.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from typing import Callable

from . import openai_client
from .client import StreamHandler, stream_message
from .config import Config
from .tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = """You are lindwyrm, a command-line coding assistant. You help \
the user work in a software project on their machine.

You have tools to read files, list directories, search (glob/grep), write and \
edit files, delete files, and run shell commands. Prefer reading before \
editing. Make minimal, targeted edits with edit_file rather than rewriting \
whole files. When running shell commands, explain what you're about to do.

Some actions require the user's confirmation and some paths are off-limits by \
policy. If a tool reports it was blocked or declined, do not retry blindly -- \
explain to the user what you wanted to do and why, and ask how to proceed.

Be concise. When you finish a task, give a short summary of what changed."""

SUMMARY_SYSTEM = """You are compacting a coding session's history so work can \
continue in a smaller context. Write a dense summary that preserves everything \
needed to keep going:

- what the user is trying to accomplish, in their own terms
- files inspected or changed, with the paths, and what changed in each
- decisions made and approaches already ruled out (and why)
- commands run and what they showed (test results, errors)
- what is still unfinished or unresolved

Leave out pleasantries and reasoning that led nowhere. Write it as notes for \
yourself, not as a report to the user. Do not invent anything that is not in \
the history."""

SUMMARY_PREFIX = (
    "[Summary of earlier conversation, compacted to save context. Treat this "
    "as established background:]\n\n"
)

# Rough fallback when a provider reports no usage numbers. English prose and
# code both land near 4 characters per token; this only needs to be close
# enough to trigger compaction at a sane time.
CHARS_PER_TOKEN = 4


def _audit(cfg: Config, record: dict) -> None:
    if not cfg.audit_log:
        return
    record["ts"] = _dt.datetime.now().isoformat(timespec="seconds")
    try:
        cfg.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with cfg.audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def estimate_tokens(messages: list[dict], system: str = "") -> int:
    """Character-based token estimate, used only when the API reports nothing."""
    chars = len(system)
    for m in messages:
        chars += len(json.dumps(m, ensure_ascii=False))
    return chars // CHARS_PER_TOKEN


def is_turn_boundary(msg: dict) -> bool:
    """True if `msg` starts a fresh user turn (not a batch of tool results).

    History may only be cut at one of these. Cutting anywhere else can orphan
    a tool_result from its tool_use -- or vice versa -- which the APIs reject.
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content", [])
    if not isinstance(content, list):
        return True
    return not any(b.get("type") == "tool_result" for b in content)


def find_cut_index(messages: list[dict], keep_last: int) -> int:
    """Index to cut history at, keeping roughly the last `keep_last` messages.

    Returns the latest turn boundary at or before the desired point, or 0 when
    no boundary is early enough to be worth cutting at.
    """
    # Clamp to a real index: keep_last=0 would otherwise start the scan one
    # past the end of the list.
    start = min(max(0, len(messages) - keep_last), len(messages) - 1)
    for i in range(start, 0, -1):
        if is_turn_boundary(messages[i]):
            return i
    return 0


class Agent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.messages: list[dict] = []
        # Input tokens the API reported for the most recent request; this is
        # what the next request will cost before anything new is added.
        self.last_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    # -- context accounting -------------------------------------------------

    def context_tokens(self) -> int:
        """Best available measure of how much context the history occupies."""
        if self.last_input_tokens > 0:
            return self.last_input_tokens
        return estimate_tokens(self.messages, SYSTEM_PROMPT)

    def context_fraction(self) -> float:
        limit = max(1, self.cfg.context_limit)
        return self.context_tokens() / limit

    def should_compact(self) -> bool:
        if not self.cfg.auto_compact or len(self.messages) < 4:
            return False
        return self.context_fraction() >= self.cfg.compact_threshold

    # -- model call ---------------------------------------------------------

    def _call_model(self, messages, system, tools, *, cfg=None, on_text=None,
                    on_thinking=None, on_retry=None):
        """Dispatch to whichever wire client the active preset uses.

        `cfg` overrides the agent's config for this one call (compaction uses
        it to turn thinking off).
        """
        cfg = cfg or self.cfg
        send = openai_client.stream_message if cfg.format == "openai" else stream_message
        return send(cfg, messages, system, tools,
                    on_text=on_text, on_thinking=on_thinking, on_retry=on_retry)

    # -- compaction ---------------------------------------------------------

    def compact(self, *, on_retry=None) -> tuple[bool, str]:
        """Summarize older history in place.

        Returns (did_compact, note). Cuts only at a user-turn boundary, so
        tool_use/tool_result pairs are never split.
        """
        cut = find_cut_index(self.messages, self.cfg.compact_keep_last)
        if cut == 0:
            return False, "nothing old enough to compact yet"

        before = self.context_tokens()
        older, recent = self.messages[:cut], self.messages[cut:]

        # Summarize with thinking off and no tools: this call is pure
        # summarization, and reasoning tokens here would be paid for twice.
        try:
            summary_cfg = dataclasses.replace(self.cfg, thinking=False)
        except TypeError:
            summary_cfg = self.cfg  # not a dataclass (tests, custom configs)
        transcript = older + [{
            "role": "user",
            "content": [{"type": "text", "text":
                         "Summarize the conversation above as instructed."}],
        }]
        try:
            handler = self._call_model(transcript, SUMMARY_SYSTEM, [],
                                       cfg=summary_cfg, on_retry=on_retry)
        except Exception as e:  # noqa: BLE001 -- compaction must never kill a session
            return False, f"compaction failed ({type(e).__name__}: {e})"

        summary = "".join(
            b.get("text", "") for b in handler.content if b.get("type") == "text"
        ).strip()
        if not summary:
            return False, "compaction failed (model returned no summary)"

        # A synthetic user/assistant pair keeps roles alternating, which the
        # Anthropic wire format expects.
        self.messages = [
            {"role": "user",
             "content": [{"type": "text", "text": SUMMARY_PREFIX + summary}]},
            {"role": "assistant",
             "content": [{"type": "text", "text":
                          "Understood -- continuing from that context."}]},
        ] + recent

        # The old count describes history that no longer exists; re-estimate so
        # the next should_compact() isn't answered with a stale number.
        self.last_input_tokens = estimate_tokens(self.messages, SYSTEM_PROMPT)
        after = self.last_input_tokens
        saved = max(0, before - after)
        return True, f"compacted {cut} message(s), ~{saved} tokens freed"

    # -- main loop ----------------------------------------------------------

    def run_turn(
        self,
        *,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str, bool], None] | None = None,
        on_retry: Callable[[int, str, float], None] | None = None,
        on_notice: Callable[[str], None] | None = None,
        max_steps: int = 50,
    ) -> None:
        """Run model<->tool cycles until the model is done (end_turn)."""
        for _ in range(max_steps):
            if self.should_compact():
                if on_notice:
                    on_notice("context is filling up, compacting history…")
                ok, note = self.compact(on_retry=on_retry)
                if on_notice:
                    on_notice(note)

            handler: StreamHandler = self._call_model(
                self.messages, SYSTEM_PROMPT, TOOL_SCHEMAS,
                on_text=on_text, on_thinking=on_thinking, on_retry=on_retry,
            )
            if handler.input_tokens:
                self.last_input_tokens = handler.input_tokens
            self.total_output_tokens += handler.output_tokens

            # Echo assistant content back verbatim (keeps thinking blocks).
            self.messages.append({"role": "assistant", "content": handler.content})

            tool_uses = [b for b in handler.content if b.get("type") == "tool_use"]
            if not tool_uses:
                return  # end_turn

            tool_results = []
            for tu in tool_uses:
                name = tu["name"]
                tool_input = tu.get("input", {})
                if on_tool:
                    on_tool(name, tool_input)
                result, is_error = run_tool(self.cfg, name, tool_input)
                _audit(self.cfg, {
                    "tool": name,
                    "input": tool_input,
                    "error": is_error,
                    "result_preview": result[:200],
                })
                if on_tool_result:
                    on_tool_result(name, result, is_error)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                    "is_error": is_error,
                })
            self.messages.append({"role": "user", "content": tool_results})
        # Hit the step ceiling.
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "(stopped: too many tool steps)"}],
        })
