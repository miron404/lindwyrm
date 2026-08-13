"""End-to-end tests against a fake provider served over real HTTP.

These exercise the parts unit tests can't reach: actual socket traffic, SSE
framing, the retry path, and a full model -> tool -> model agent loop. The
"provider" is a stdlib http.server on a random port, so nothing leaves the
machine and no API key is involved.
"""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lindwyrm import http as lw_http
from lindwyrm.agent import Agent
from lindwyrm.config import Config, Policy


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def anthropic_stream(text=None, tool=None, input_tokens=100, output_tokens=10):
    """Build an Anthropic-format SSE response body."""
    out = [sse("message_start", {"message": {"usage": {
        "input_tokens": input_tokens, "output_tokens": 0}}})]
    if text is not None:
        out += [
            sse("content_block_start", {"index": 0,
                                        "content_block": {"type": "text", "text": ""}}),
            sse("content_block_delta", {"index": 0,
                                        "delta": {"type": "text_delta", "text": text}}),
            sse("content_block_stop", {"index": 0}),
        ]
    if tool is not None:
        name, args = tool
        out += [
            sse("content_block_start", {"index": 1, "content_block": {
                "type": "tool_use", "id": "tu_1", "name": name}}),
            sse("content_block_delta", {"index": 1, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(args)}}),
            sse("content_block_stop", {"index": 1}),
        ]
    out.append(sse("message_delta", {"delta": {"stop_reason": "end_turn"},
                                     "usage": {"output_tokens": output_tokens}}))
    out.append(sse("message_stop", {}))
    return b"".join(out)


class FakeProvider(BaseHTTPRequestHandler):
    """Replies by popping from the class-level `script`."""

    script: list = []
    requests: list = []

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        FakeProvider.requests.append(json.loads(body or b"{}"))

        status, payload = FakeProvider.script.pop(0)
        self.send_response(status)
        if status == 200:
            self.send_header("content-type", "text/event-stream")
        else:
            self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass  # keep test output clean


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        FakeProvider.script = []
        FakeProvider.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
        # shutdown() blocks until the poll loop notices; the 0.5s default would
        # add half a second to every single test.
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01),
            daemon=True,
        )
        self.thread.start()
        port = self.server.server_address[1]

        # Retries must not actually sleep in tests.
        self._real_sleep = lw_http.time.sleep
        self.slept = []
        lw_http.time.sleep = self.slept.append

        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{port}",
            model="fake-model",
            format="anthropic",
            thinking=False,
            project_root=Path(self._tmp.name).resolve(),
            policy=Policy(read="allow", write="allow"),
        )

    def tearDown(self):
        lw_http.time.sleep = self._real_sleep
        lw_http.close_client()
        self.server.shutdown()
        self.server.server_close()
        self._tmp.cleanup()


class TestStreaming(ProviderTestCase):
    def test_plain_answer_streams_through(self):
        FakeProvider.script = [(200, anthropic_stream(text="hello world"))]
        agent = Agent(self.cfg)
        agent.add_user("hi")
        chunks = []
        agent.run_turn(on_text=chunks.append)

        self.assertEqual("".join(chunks), "hello world")
        self.assertEqual(agent.messages[-1]["role"], "assistant")

    def test_usage_is_captured_from_the_stream(self):
        FakeProvider.script = [(200, anthropic_stream(text="x", input_tokens=4242,
                                                      output_tokens=17))]
        agent = Agent(self.cfg)
        agent.add_user("hi")
        agent.run_turn(on_text=lambda _: None)

        self.assertEqual(agent.last_input_tokens, 4242)
        self.assertEqual(agent.total_output_tokens, 17)

    def test_system_prompt_and_tools_are_sent(self):
        FakeProvider.script = [(200, anthropic_stream(text="ok"))]
        agent = Agent(self.cfg)
        agent.add_user("hi")
        agent.run_turn(on_text=lambda _: None)

        sent = FakeProvider.requests[0]
        self.assertIn("lindwyrm", sent["system"])
        self.assertTrue(any(t["name"] == "read_file" for t in sent["tools"]))


class TestRetry(ProviderTestCase):
    def test_429_is_retried_then_succeeds(self):
        FakeProvider.script = [
            (429, b'{"error":"slow down"}'),
            (200, anthropic_stream(text="recovered")),
        ]
        agent = Agent(self.cfg)
        agent.add_user("hi")
        seen = []
        agent.run_turn(on_text=seen.append,
                       on_retry=lambda a, r, d: seen.append(f"[retry {r}]"))

        self.assertIn("[retry HTTP 429]", seen)
        self.assertIn("recovered", seen)
        self.assertEqual(len(self.slept), 1)

    def test_500_is_retried(self):
        FakeProvider.script = [
            (500, b'{"error":"boom"}'),
            (200, anthropic_stream(text="ok")),
        ]
        agent = Agent(self.cfg)
        agent.add_user("hi")
        agent.run_turn(on_text=lambda _: None)
        self.assertEqual(len(FakeProvider.requests), 2)

    def test_400_is_not_retried(self):
        """A bad request fails the same way every time; retrying wastes time."""
        FakeProvider.script = [(400, b'{"error":"bad request"}')]
        agent = Agent(self.cfg)
        agent.add_user("hi")
        with self.assertRaises(lw_http.APIError):
            agent.run_turn(on_text=lambda _: None)
        self.assertEqual(len(FakeProvider.requests), 1)
        self.assertEqual(self.slept, [])

    def test_gives_up_after_max_retries(self):
        self.cfg.max_retries = 3
        FakeProvider.script = [(503, b"nope")] * 3
        agent = Agent(self.cfg)
        agent.add_user("hi")
        with self.assertRaises(lw_http.APIError):
            agent.run_turn(on_text=lambda _: None)
        self.assertEqual(len(FakeProvider.requests), 3)


class TestAgentLoop(ProviderTestCase):
    def test_tool_call_round_trip(self):
        """model asks for a tool -> tool runs -> result goes back -> final answer."""
        (Path(self.cfg.project_root) / "hello.txt").write_text("file contents here")

        FakeProvider.script = [
            (200, anthropic_stream(tool=("read_file", {"path": "hello.txt"}))),
            (200, anthropic_stream(text="the file says hello")),
        ]
        agent = Agent(self.cfg)
        agent.add_user("read hello.txt")
        tools_run, texts = [], []
        agent.run_turn(on_text=texts.append,
                       on_tool=lambda n, i: tools_run.append(n))

        self.assertEqual(tools_run, ["read_file"])
        self.assertEqual("".join(texts), "the file says hello")

        # The second request must carry the tool result back to the model.
        second = FakeProvider.requests[1]
        results = [b for m in second["messages"] if m["role"] == "user"
                   for b in m["content"] if b.get("type") == "tool_result"]
        self.assertEqual(len(results), 1)
        self.assertIn("file contents here", results[0]["content"])

    def test_tool_error_is_reported_back_not_raised(self):
        FakeProvider.script = [
            (200, anthropic_stream(tool=("read_file", {"path": "missing.txt"}))),
            (200, anthropic_stream(text="that file does not exist")),
        ]
        agent = Agent(self.cfg)
        agent.add_user("read missing.txt")
        errors = []
        agent.run_turn(on_text=lambda _: None,
                       on_tool_result=lambda n, r, e: errors.append(e))

        self.assertEqual(errors, [True])
        second = FakeProvider.requests[1]
        result = [b for m in second["messages"] if m["role"] == "user"
                  for b in m["content"] if b.get("type") == "tool_result"][0]
        self.assertTrue(result["is_error"])

    def test_write_tool_actually_writes(self):
        FakeProvider.script = [
            (200, anthropic_stream(tool=("write_file",
                                         {"path": "out.txt", "content": "written!"}))),
            (200, anthropic_stream(text="done")),
        ]
        agent = Agent(self.cfg)
        agent.add_user("write out.txt")
        agent.run_turn(on_text=lambda _: None)

        self.assertEqual(
            (Path(self.cfg.project_root) / "out.txt").read_text(), "written!")


class TestCompactionOverHttp(ProviderTestCase):
    def test_auto_compaction_fires_and_history_stays_valid(self):
        self.cfg.context_limit = 1000
        self.cfg.compact_threshold = 0.5
        self.cfg.compact_keep_last = 2

        agent = Agent(self.cfg)
        agent.messages = [
            {"role": "user", "content": [{"type": "text", "text": "old q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old a"}]},
            {"role": "user", "content": [{"type": "text", "text": "new q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "new a"}]},
        ]
        agent.last_input_tokens = 900  # 90% full -> must compact

        FakeProvider.script = [
            (200, anthropic_stream(text="SUMMARY")),   # the compaction call
            (200, anthropic_stream(text="answer")),    # the real turn
        ]
        agent.add_user("next question")
        notices = []
        agent.run_turn(on_text=lambda _: None, on_notice=notices.append)

        self.assertTrue(any("compact" in n for n in notices), notices)
        self.assertIn("SUMMARY", json.dumps(agent.messages))

        # The turn request must still have alternating, well-formed roles.
        roles = [m["role"] for m in FakeProvider.requests[1]["messages"]]
        for a, b in zip(roles, roles[1:]):
            self.assertNotEqual(a, b, f"consecutive {a} in {roles}")

    def test_compaction_call_carries_no_tools(self):
        self.cfg.context_limit = 1000
        self.cfg.compact_threshold = 0.5
        self.cfg.compact_keep_last = 2
        agent = Agent(self.cfg)
        agent.messages = [
            {"role": "user", "content": [{"type": "text", "text": "q1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": [{"type": "text", "text": "q2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
        ]
        agent.last_input_tokens = 900
        FakeProvider.script = [
            (200, anthropic_stream(text="SUMMARY")),
            (200, anthropic_stream(text="answer")),
        ]
        agent.add_user("go")
        agent.run_turn(on_text=lambda _: None)

        self.assertEqual(FakeProvider.requests[0].get("tools"), [])
        self.assertTrue(FakeProvider.requests[1]["tools"])


if __name__ == "__main__":
    unittest.main()
