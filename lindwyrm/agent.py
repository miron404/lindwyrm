"""The agentic loop: model -> tool calls -> tool results -> model -> ...

Keeps the full message history (including thinking blocks, which DeepSeek
requires echoed back). Stops when the model returns end_turn without
requesting more tools.
"""

from __future__ import annotations

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


class Agent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def run_turn(
        self,
        *,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str, bool], None] | None = None,
        max_steps: int = 50,
    ) -> None:
        """Run model<->tool cycles until the model is done (end_turn)."""
        for _ in range(max_steps):
            if self.cfg.format == "openai":
                handler = openai_client.stream_message(
                    self.cfg,
                    self.messages,
                    SYSTEM_PROMPT,
                    TOOL_SCHEMAS,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )
            else:
                handler: StreamHandler = stream_message(
                    self.cfg,
                    self.messages,
                    SYSTEM_PROMPT,
                    TOOL_SCHEMAS,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )
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
