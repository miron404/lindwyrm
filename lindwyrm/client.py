"""Minimal client for the DeepSeek Anthropic-compatible Messages API.

We deliberately do NOT use the official `anthropic` SDK -- fewer dependencies,
nothing that could phone home, and full control over the request body. Just
httpx + the documented wire format.

Key DeepSeek-specific details baked in here:
  * system is a top-level field, not a message.
  * thinking is on by default; we send thinking:{type:"enabled", budget_tokens}
    or thinking:{type:"disabled"}.
  * The assistant's thinking blocks MUST be echoed back in message history on
    subsequent turns, or the API returns 400. The agent loop preserves the
    full content array, so this is handled by passing assistant content back
    verbatim.
  * Auth via x-api-key + anthropic-version header.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from .config import ANTHROPIC_VERSION, Config


class APIError(Exception):
    pass


def _headers(cfg: Config) -> dict[str, str]:
    return {
        "x-api-key": cfg.api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _build_body(cfg: Config, messages: list[dict], system: str, tools: list[dict]) -> dict:
    body: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    if cfg.thinking:
        # budget must be < max_tokens; clamp defensively.
        budget = min(cfg.thinking_budget, max(1024, cfg.max_tokens - 1024))
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        body["thinking"] = {"type": "disabled"}
        # temperature is only meaningful with thinking disabled.
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
    return body


class StreamHandler:
    """Accumulates a streamed response into a content-block array.

    Yields ('text', str) and ('thinking', str) deltas as they arrive so the
    caller can print them live. After iteration, `.content` holds the full
    assistant content array ready to append to message history, and
    `.stop_reason` holds the stop reason.
    """

    def __init__(self) -> None:
        self.content: list[dict] = []
        self.stop_reason: str | None = None
        self._blocks: dict[int, dict] = {}

    def feed(self, event_type: str, data: dict) -> Iterator[tuple[str, str]]:
        if event_type == "content_block_start":
            idx = data["index"]
            block = data["content_block"]
            self._blocks[idx] = block
            if block.get("type") == "tool_use":
                block.setdefault("input", {})
                block["_partial_json"] = ""
        elif event_type == "content_block_delta":
            idx = data["index"]
            delta = data["delta"]
            block = self._blocks.get(idx, {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                block.setdefault("text", "")
                block["text"] += delta["text"]
                yield ("text", delta["text"])
            elif dtype == "thinking_delta":
                block.setdefault("thinking", "")
                block["thinking"] += delta["thinking"]
                yield ("thinking", delta["thinking"])
            elif dtype == "signature_delta":
                block.setdefault("signature", "")
                block["signature"] += delta.get("signature", "")
            elif dtype == "input_json_delta":
                block["_partial_json"] += delta.get("partial_json", "")
        elif event_type == "content_block_stop":
            idx = data["index"]
            block = self._blocks.get(idx)
            if block and block.get("type") == "tool_use":
                raw = block.pop("_partial_json", "")
                try:
                    block["input"] = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    block["input"] = {}
        elif event_type == "message_delta":
            self.stop_reason = data.get("delta", {}).get("stop_reason", self.stop_reason)
        elif event_type == "message_stop":
            # Finalize content in index order, stripping internal scratch keys.
            self.content = []
            for i in sorted(self._blocks):
                b = dict(self._blocks[i])
                b.pop("_partial_json", None)
                self.content.append(b)


def stream_message(
    cfg: Config,
    messages: list[dict],
    system: str,
    tools: list[dict],
    *,
    on_text=None,
    on_thinking=None,
) -> StreamHandler:
    """Send a streaming request and return the completed StreamHandler.

    on_text / on_thinking are optional callbacks(str) for live printing.
    """
    body = _build_body(cfg, messages, system, tools)
    body["stream"] = True
    handler = StreamHandler()
    url = cfg.base_url.rstrip("/") + "/v1/messages"

    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        with client.stream("POST", url, headers=_headers(cfg), json=body) as resp:
            if resp.status_code != 200:
                detail = resp.read().decode("utf-8", errors="replace")
                raise APIError(f"HTTP {resp.status_code}: {detail}")
            event_type = None
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw if isinstance(raw, str) else raw.decode("utf-8")
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event_type == "error":
                        raise APIError(f"Stream error: {data}")
                    for kind, chunk in handler.feed(event_type, data):
                        if kind == "text" and on_text:
                            on_text(chunk)
                        elif kind == "thinking" and on_thinking:
                            on_thinking(chunk)
    return handler
