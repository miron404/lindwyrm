"""The history must stay a shape both APIs accept, whatever a turn runs into.

Each case here once left a session that answered every later message with a
400: a tool batch stopped by [q]uit or Ctrl+C (tool_use without tool_result),
a stream that died before message_stop (an empty assistant message), and a
reply cut off by max_tokens mid tool call (arguments silently became {}).
"""

import unittest
from unittest import mock

from lindwyrm import agent as agent_mod
from lindwyrm.agent import Agent
from lindwyrm.client import StreamHandler, stream_message
from lindwyrm.http import APIError
from lindwyrm.sandbox import UserQuit

from helpers import make_config


def reply(*blocks, stop="end_turn", bad=()):
    h = StreamHandler()
    h.content = list(blocks)
    h.stop_reason = stop
    h.complete = True
    h.bad_tool_input = set(bad)
    return h


def tool(id_, name="read_file", **args):
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def text(t):
    return {"type": "text", "text": t}


def paired(messages):
    """True if every tool_use is answered in the message right after it."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
        if not ids:
            continue
        if i + 1 >= len(messages):
            return False
        answered = {b.get("tool_use_id") for b in messages[i + 1]["content"]
                    if b.get("type") == "tool_result"}
        if not ids <= answered:
            return False
    return True


class TurnCase(unittest.TestCase):
    def make_agent(self, *replies):
        a = Agent(make_config())
        script = list(replies)
        a._call_model = lambda *args, **kw: script.pop(0)
        a.add_user("go")
        return a


class TestInterruptedBatch(TurnCase):
    def test_quit_at_a_prompt_still_answers_every_tool_use(self):
        a = self.make_agent(reply(tool("t1"), tool("t2")))
        with mock.patch.object(agent_mod, "run_tool", side_effect=UserQuit):
            with self.assertRaises(UserQuit):
                a.run_turn(on_text=lambda _: None)
        self.assertTrue(paired(a.messages))
        results = a.messages[-1]["content"]
        self.assertEqual([r["content"] for r in results],
                         [agent_mod.INTERRUPTED_RESULT] * 2)
        self.assertTrue(all(r["is_error"] for r in results))

    def test_results_gathered_before_ctrl_c_are_kept(self):
        a = self.make_agent(reply(tool("t1"), tool("t2")))
        outcomes = iter([("real output", False), KeyboardInterrupt()])

        def fake_run_tool(*args, **kw):
            out = next(outcomes)
            if isinstance(out, BaseException):
                raise out
            return out

        with mock.patch.object(agent_mod, "run_tool", side_effect=fake_run_tool):
            with self.assertRaises(KeyboardInterrupt):
                a.run_turn(on_text=lambda _: None)
        results = a.messages[-1]["content"]
        self.assertEqual(results[0]["content"], "real output")
        self.assertEqual(results[1]["content"], agent_mod.INTERRUPTED_RESULT)
        self.assertTrue(paired(a.messages))


class TestMaxTokens(TurnCase):
    def test_a_cut_off_tool_call_is_not_run(self):
        a = self.make_agent(
            reply(text("writing it"), tool("t1", "write_file"), stop="max_tokens"),
            reply(text("ok, smaller then")))
        with mock.patch.object(agent_mod, "run_tool") as run:
            self.assertTrue(a.run_turn(on_text=lambda _: None))
        run.assert_not_called()
        result = a.messages[2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("max_tokens", result["content"])
        self.assertTrue(paired(a.messages))

    def test_complete_calls_before_the_cut_one_still_run(self):
        a = self.make_agent(
            reply(tool("t1", path="a.py"), tool("t2", "write_file"), stop="max_tokens"),
            reply(text("done")))
        with mock.patch.object(agent_mod, "run_tool", return_value=("ok", False)) as run:
            a.run_turn(on_text=lambda _: None)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[1], "read_file")

    def test_unparseable_arguments_are_reported_not_run(self):
        a = self.make_agent(reply(tool("t1"), bad={"t1"}), reply(text("done")))
        with mock.patch.object(agent_mod, "run_tool") as run:
            a.run_turn(on_text=lambda _: None)
        run.assert_not_called()
        self.assertEqual(a.messages[2]["content"][0]["content"],
                         agent_mod.BAD_INPUT_RESULT)

    def test_a_truncated_answer_is_flagged(self):
        a = self.make_agent(reply(text("1. apple\n2. pea"), stop="max_tokens"))
        self.assertTrue(a.run_turn(on_text=lambda _: None))
        self.assertTrue(a.cut_off)

    def test_a_finished_answer_is_not_flagged(self):
        a = self.make_agent(reply(text("done")))
        a.run_turn(on_text=lambda _: None)
        self.assertFalse(a.cut_off)


class TestEmptyReply(TurnCase):
    def test_an_empty_reply_is_not_stored(self):
        """Both APIs reject an empty assistant message in history."""
        a = self.make_agent(reply())
        self.assertTrue(a.run_turn(on_text=lambda _: None))
        self.assertEqual([m["role"] for m in a.messages], ["user"])


def sse_events(*events):
    return mock.patch("lindwyrm.client.stream_sse", return_value=iter(events))


START = ("message_start", {"message": {"usage": {"input_tokens": 5}}})


class TestStreamCompleteness(unittest.TestCase):
    def test_a_stream_cut_before_message_stop_is_an_error(self):
        events = [START,
                  ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
                  ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Half an answ"}})]
        with sse_events(*events):
            with self.assertRaises(APIError):
                stream_message(make_config(), [], "", [])

    def test_bad_tool_json_is_recorded(self):
        events = [START,
                  ("content_block_start", {"index": 0, "content_block": {
                      "type": "tool_use", "id": "t1", "name": "write_file"}}),
                  ("content_block_delta", {"index": 0, "delta": {
                      "type": "input_json_delta", "partial_json": '{"path": "a.md", "conte'}}),
                  ("content_block_stop", {"index": 0}),
                  ("message_delta", {"delta": {"stop_reason": "max_tokens"}}),
                  ("message_stop", {})]
        with sse_events(*events):
            h = stream_message(make_config(), [], "", [])
        self.assertEqual(h.bad_tool_input, {"t1"})
        self.assertEqual(h.content[0]["input"], {})
        self.assertNotIn("_partial_json", h.content[0])

    def test_a_block_without_its_stop_event_is_still_parsed(self):
        events = [START,
                  ("content_block_start", {"index": 0, "content_block": {
                      "type": "tool_use", "id": "t1", "name": "read_file"}}),
                  ("content_block_delta", {"index": 0, "delta": {
                      "type": "input_json_delta", "partial_json": '{"path": "a.py"}'}}),
                  ("message_stop", {})]
        with sse_events(*events):
            h = stream_message(make_config(), [], "", [])
        self.assertEqual(h.content[0]["input"], {"path": "a.py"})
        self.assertNotIn("_partial_json", h.content[0])


if __name__ == "__main__":
    unittest.main()
