"""Client for OpenAI-compatible Chat Completions APIs.

Used for any preset with format="openai" (custom providers the user adds --
OpenRouter-style aggregators, Kimi, or any other OpenAI-compatible endpoint).

We deliberately do NOT use the official `openai` SDK -- same reasoning as the
Anthropic client: fewer dependencies, full control over the request, nothing
that could phone home. Just httpx + the documented wire format.

The rest of lindwyrm (agent.py, tools.py, render.py) works in a single internal
message representation: Anthropic-style content blocks (text / thinking /
tool_use / tool_result). This module is a translator at the edges:
  * to_openai_messages()/to_openai_tools() convert outbound.
  * OpenAIStreamHandler assembles an inbound SSE stream back into that same
    internal block format, so the agent loop never needs to know which wire
    format is in use.

Notable differences from the Anthropic client:
  * system is a message (role="system"), not a top-level field.
  * Tool results are separate {"role": "tool", "tool_call_id": ...} messages,
    one per result, rather than bundled into a user turn.
  * "Thinking" isn't standardized. Some OpenAI-compatible providers stream a
    non-standard `reasoning_content` delta; we surface it live (on_thinking)
    for display, but do NOT persist it into history or echo it back --
    unlike DeepSeek's Anthropic-format endpoint, OpenAI-style APIs don't
    require or expect it, and echoing an unknown field back can error on some
    providers.
  * Auth via a standard `Authorization: Bearer <key>` header.
"""

from __future__ import annotations

import json
from typing import Any

from .config import Config
from .http import APIError, stream_sse  # shared exception type across both clients


def _headers(cfg: Config) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {cfg.api_key}",
        "content-type": "application/json",
    }


# ---------------------------------------------------------------------------
# Outbound translation: internal blocks -> OpenAI wire format
# ---------------------------------------------------------------------------

def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role")
        content = m.get("content", [])
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]

        if role == "user":
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            if tool_results:
                for tr in tool_results:
                    out.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", ""),
                    })
                continue
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            out.append({"role": "user", "content": text})

        elif role == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text if text else None}
            if tool_uses:
                msg["tool_calls"] = [
                    {
                        "id": tu.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                    for tu in tool_uses
                ]
            out.append(msg)
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _build_body(cfg: Config, messages: list[dict], tools: list[dict]) -> dict:
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    if cfg.max_tokens:
        # OpenAI deprecated max_tokens in favour of max_completion_tokens and
        # newer reasoning models reject the old field outright -- while plenty
        # of OpenAI-compatible servers only understand the old one. Which to
        # send is a per-provider fact, so it's a preset option.
        field = "max_completion_tokens" if cfg.max_completion_tokens else "max_tokens"
        body[field] = cfg.max_tokens
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.extra_body:
        body.update(cfg.extra_body)
    return body


# ---------------------------------------------------------------------------
# Inbound translation: OpenAI SSE deltas -> internal blocks
# ---------------------------------------------------------------------------

_FINISH_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


class OpenAIStreamHandler:
    """Accumulates a streamed Chat Completions response into internal blocks.

    After the stream ends, call finalize(); `.content` then holds the same
    block shape the Anthropic client produces (text / tool_use), ready to
    append to message history, and `.stop_reason` mirrors Anthropic's values.
    """

    def __init__(self) -> None:
        self.content: list[dict] = []
        self.stop_reason: str | None = None
        # Real token counts, from the usage-only chunk that stream_options
        # asks the provider to send. Left at 0 by providers that ignore it.
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._text = ""
        self._tool_calls: dict[int, dict] = {}  # index -> {id, name, args}

    def feed_chunk(self, data: dict) -> tuple[str | None, str | None]:
        """Process one SSE chunk. Returns (text_delta, thinking_delta), either
        of which may be None."""
        usage = data.get("usage")
        if usage:
            self.input_tokens = int(usage.get("prompt_tokens", 0))
            self.output_tokens = int(usage.get("completion_tokens", 0))

        choices = data.get("choices") or []
        if not choices:
            return None, None  # e.g. a trailing usage-only chunk
        choice = choices[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        text_out = None
        think_out = None

        piece = delta.get("content")
        if piece:
            self._text += piece
            text_out = piece

        # Non-standard but used by some providers (e.g. reasoning models).
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            think_out = reasoning

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = self._tool_calls.setdefault(idx, {"id": None, "name": None, "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]

        if finish:
            self.stop_reason = _FINISH_MAP.get(finish, finish)

        return text_out, think_out

    def finalize(self) -> None:
        if self._text:
            self.content.append({"type": "text", "text": self._text})
        for idx in sorted(self._tool_calls):
            tc = self._tool_calls[idx]
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {}
            self.content.append({
                "type": "tool_use",
                "id": tc["id"] or f"call_{idx}",
                "name": tc["name"] or "",
                "input": args,
            })
        if self.stop_reason is None:
            self.stop_reason = "end_turn"


def stream_message(
    cfg: Config,
    messages: list[dict],
    system: str,
    tools: list[dict],
    *,
    on_text=None,
    on_thinking=None,
    on_retry=None,
) -> OpenAIStreamHandler:
    """Send a streaming chat.completions request and return the completed
    handler. Same calling convention as client.stream_message (Anthropic)."""
    body = _build_body(cfg, to_openai_messages(system, messages), to_openai_tools(tools))
    handler = OpenAIStreamHandler()
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    for _event, data in stream_sse(url, _headers(cfg), body,
                                   proxy=cfg.proxy,
                                   max_attempts=cfg.max_retries,
                                   on_retry=on_retry):
        if "error" in data:
            raise APIError(f"Stream error: {data['error']}")
        text_out, think_out = handler.feed_chunk(data)
        if text_out and on_text:
            on_text(text_out)
        if think_out and on_thinking:
            on_thinking(think_out)

    handler.finalize()
    return handler
