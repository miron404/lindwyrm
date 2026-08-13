"""Shared HTTP plumbing for both wire clients.

Three things live here that used to be duplicated (or missing) in client.py and
openai_client.py:

  * A reused httpx.Client. Building one per request meant a fresh TLS handshake
    on every single turn of a conversation; keeping one alive reuses the
    connection instead.
  * Retry with exponential backoff for transient failures (429 and 5xx, plus
    connect/read errors). An agent loop can issue dozens of requests per task,
    so a single blip used to kill a whole turn's work.
  * SSE line parsing, which both wire formats share even though the event
    payloads inside differ.

Retry safety rule: we only retry a request that failed BEFORE any bytes of the
response body were handed to the caller. Once text has been streamed to the
screen, replaying the request would duplicate that output, so at that point an
error propagates instead.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Iterator

import httpx

# Status codes worth retrying: rate limits and transient server-side faults.
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE = 1.0  # seconds; doubled each attempt
MAX_BACKOFF = 30.0

_client: httpx.Client | None = None


class APIError(Exception):
    pass


def get_client() -> httpx.Client:
    """Return the process-wide httpx.Client, creating it on first use."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=30.0),
            # An agent talks to one host at a time; a small pool is plenty.
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _client


def close_client() -> None:
    """Close the shared client. Safe to call more than once."""
    global _client
    if _client is not None and not _client.is_closed:
        _client.close()
    _client = None


def backoff_delay(attempt: int, retry_after: float | None = None,
                  base: float = DEFAULT_BACKOFF_BASE) -> float:
    """Seconds to wait before retry number `attempt` (1-based).

    Honors a server-provided Retry-After when there is one, otherwise doubles
    each attempt with jitter so several clients don't retry in lockstep.
    """
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, MAX_BACKOFF)
    delay = min(base * (2 ** (attempt - 1)), MAX_BACKOFF)
    return delay * (0.5 + random.random() * 0.5)  # jitter: 50-100% of delay


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None  # HTTP-date form; fall back to computed backoff


def stream_sse(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
    on_retry: Callable[[int, str, float], None] | None = None,
) -> Iterator[tuple[str | None, dict]]:
    """POST `body` and yield (event_type, parsed_data) for each SSE data line.

    `event_type` comes from a preceding `event:` line when the server sends one
    (Anthropic does; OpenAI does not, leaving it None).

    on_retry(attempt, reason, delay) is called before each backoff sleep so the
    CLI can tell the user why it's pausing.
    """
    # Resolved here rather than as a default argument, which would bind
    # time.sleep at import time and make the delay unpatchable.
    if sleep is None:
        sleep = time.sleep
    client = get_client()
    last_error: str = "unknown error"

    for attempt in range(1, max_attempts + 1):
        try:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    detail = resp.read().decode("utf-8", errors="replace")
                    if resp.status_code in RETRY_STATUS and attempt < max_attempts:
                        last_error = f"HTTP {resp.status_code}"
                        delay = backoff_delay(attempt, _retry_after_seconds(resp))
                        if on_retry:
                            on_retry(attempt, last_error, delay)
                        sleep(delay)
                        continue
                    raise APIError(f"HTTP {resp.status_code}: {detail}")

                # Status is good; from here on the caller may start seeing
                # output, so a failure must not be retried.
                yield from _iter_events(resp)
                return

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.RemoteProtocolError) as e:
            if attempt >= max_attempts:
                raise APIError(f"{type(e).__name__}: {e}") from e
            last_error = type(e).__name__
            delay = backoff_delay(attempt)
            if on_retry:
                on_retry(attempt, last_error, delay)
            sleep(delay)

    raise APIError(f"Giving up after {max_attempts} attempts ({last_error}).")


def _iter_events(resp: httpx.Response) -> Iterator[tuple[str | None, dict]]:
    event_type: str | None = None
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            yield event_type, data
