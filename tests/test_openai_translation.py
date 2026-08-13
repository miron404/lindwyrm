"""The internal block format <-> OpenAI wire format translation."""

import json
import unittest

from lindwyrm.openai_client import (
    OpenAIStreamHandler,
    to_openai_messages,
    to_openai_tools,
)


class TestOutboundMessages(unittest.TestCase):
    def test_system_goes_first_as_a_message(self):
        out = to_openai_messages("SYS", [])
        self.assertEqual(out[0], {"role": "system", "content": "SYS"})

    def test_user_text_is_flattened(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        out = to_openai_messages("s", msgs)
        self.assertEqual(out[1], {"role": "user", "content": "ab"})

    def test_tool_use_becomes_tool_calls(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "read_file",
             "input": {"path": "a.py"}}]}]
        out = to_openai_messages("s", msgs)
        call = out[1]["tool_calls"][0]
        self.assertEqual(call["id"], "t1")
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a.py"})

    def test_tool_results_become_separate_tool_messages(self):
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "out1"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "out2"}]}]
        out = to_openai_messages("s", msgs)[1:]
        self.assertEqual([m["role"] for m in out], ["tool", "tool"])
        self.assertEqual([m["tool_call_id"] for m in out], ["t1", "t2"])

    def test_thinking_blocks_are_dropped(self):
        """Reasoning is not echoed back in OpenAI format -- some providers
        reject unknown fields."""
        msgs = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "answer"}]}]
        out = to_openai_messages("s", msgs)
        self.assertEqual(out[1]["content"], "answer")
        self.assertNotIn("thinking", json.dumps(out))

    def test_assistant_with_only_tool_calls_has_null_content(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "n", "input": {}}]}]
        self.assertIsNone(to_openai_messages("s", msgs)[1]["content"])


class TestOutboundTools(unittest.TestCase):
    def test_schema_shape(self):
        got = to_openai_tools([{
            "name": "read_file", "description": "d",
            "input_schema": {"type": "object", "properties": {}},
        }])[0]
        self.assertEqual(got["type"], "function")
        self.assertEqual(got["function"]["name"], "read_file")
        self.assertEqual(got["function"]["parameters"]["type"], "object")


class TestInboundStream(unittest.TestCase):
    def feed_all(self, chunks):
        h = OpenAIStreamHandler()
        for c in chunks:
            h.feed_chunk(c)
        h.finalize()
        return h

    def test_text_accumulates_into_one_block(self):
        h = self.feed_all([
            {"choices": [{"delta": {"content": "he"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        self.assertEqual(h.content, [{"type": "text", "text": "hello"}])
        self.assertEqual(h.stop_reason, "end_turn")

    def test_tool_call_arguments_are_reassembled(self):
        h = self.feed_all([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "grep", "arguments": '{"pat'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'tern":"x"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])
        block = h.content[0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["name"], "grep")
        self.assertEqual(block["input"], {"pattern": "x"})
        self.assertEqual(h.stop_reason, "tool_use")

    def test_malformed_tool_arguments_do_not_crash(self):
        h = self.feed_all([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "g", "arguments": "{bad"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])
        self.assertEqual(h.content[0]["input"], {})

    def test_reasoning_is_surfaced_but_not_stored(self):
        h = OpenAIStreamHandler()
        _text, think = h.feed_chunk(
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]})
        h.finalize()
        self.assertEqual(think, "thinking...")
        self.assertEqual(h.content, [])  # not echoed back into history

    def test_usage_chunk_is_recorded(self):
        h = self.feed_all([
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 7}},
        ])
        self.assertEqual(h.input_tokens, 120)
        self.assertEqual(h.output_tokens, 7)

    def test_usage_only_chunk_without_choices_is_safe(self):
        h = OpenAIStreamHandler()
        self.assertEqual(h.feed_chunk({"choices": []}), (None, None))

    def test_missing_finish_reason_defaults_to_end_turn(self):
        h = self.feed_all([{"choices": [{"delta": {"content": "x"}}]}])
        self.assertEqual(h.stop_reason, "end_turn")


if __name__ == "__main__":
    unittest.main()
