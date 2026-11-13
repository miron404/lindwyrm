"""History compaction: cutting in the wrong place is an API 400, so the
boundary logic is the part worth pinning down."""

import unittest

from helpers import make_config

from lindwyrm.agent import (
    REQUEST_MARKER,
    SUMMARY_PREFIX,
    Agent,
    estimate_tokens,
    find_cut_index,
    is_turn_boundary,
    midturn_cut_at_or_before,
)
from lindwyrm.session import _title_from


def user_text(text="hi"):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_text(text="ok"):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def assistant_tool_use(tool_id="t1"):
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": tool_id, "name": "read_file", "input": {}},
    ]}


def tool_results(tool_id="t1"):
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": "..."},
    ]}


class TestTurnBoundary(unittest.TestCase):
    def test_plain_user_message_is_a_boundary(self):
        self.assertTrue(is_turn_boundary(user_text()))

    def test_tool_results_are_not_a_boundary(self):
        self.assertFalse(is_turn_boundary(tool_results()))

    def test_assistant_messages_are_never_a_boundary(self):
        self.assertFalse(is_turn_boundary(assistant_text()))
        self.assertFalse(is_turn_boundary(assistant_tool_use()))

    def test_non_list_content_is_treated_as_a_boundary(self):
        self.assertTrue(is_turn_boundary({"role": "user", "content": "plain string"}))


class TestFindCutIndex(unittest.TestCase):
    def test_never_splits_a_tool_use_from_its_result(self):
        # user, assistant(tool_use), user(tool_result), assistant, user, assistant
        msgs = [
            user_text("first"),
            assistant_tool_use(),
            tool_results(),
            assistant_text(),
            user_text("second"),
            assistant_text(),
        ]
        for keep in range(0, len(msgs) + 2):
            cut = find_cut_index(msgs, keep)
            if cut:
                self.assertTrue(
                    is_turn_boundary(msgs[cut]),
                    f"keep_last={keep} cut at {cut}, which is not a turn boundary",
                )

    def test_returns_zero_when_no_boundary_is_early_enough(self):
        # A single long turn: the only boundary is index 0, which means
        # "nothing to compact".
        msgs = [user_text(), assistant_tool_use(), tool_results(), assistant_text()]
        self.assertEqual(find_cut_index(msgs, keep_last=4), 0)

    def test_cuts_at_the_latest_boundary_within_budget(self):
        msgs = [
            user_text("a"), assistant_text(),
            user_text("b"), assistant_text(),
            user_text("c"), assistant_text(),
        ]
        # Keeping the last 2 messages -> the boundary at index 4 ("c").
        self.assertEqual(find_cut_index(msgs, keep_last=2), 4)

    def test_keeping_everything_cuts_nothing(self):
        msgs = [user_text("a"), assistant_text(), user_text("b")]
        self.assertEqual(find_cut_index(msgs, keep_last=len(msgs)), 0)

    def test_empty_history(self):
        self.assertEqual(find_cut_index([], keep_last=4), 0)

    def test_cut_keeps_history_replayable(self):
        """Whatever survives a cut must still start a turn, so the rebuilt
        history stays valid when the summary pair is prepended."""
        msgs = [
            user_text("a"), assistant_tool_use("t1"), tool_results("t1"),
            assistant_text(),
            user_text("b"), assistant_tool_use("t2"), tool_results("t2"),
            assistant_text(),
        ]
        cut = find_cut_index(msgs, keep_last=3)
        self.assertGreater(cut, 0)
        remaining = msgs[cut:]
        self.assertTrue(is_turn_boundary(remaining[0]))
        # No tool_result may reference a tool_use that was cut away.
        kept_ids = {
            b["id"]
            for m in remaining if m["role"] == "assistant"
            for b in m["content"] if b.get("type") == "tool_use"
        }
        for m in remaining:
            if m["role"] != "user":
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    self.assertIn(b["tool_use_id"], kept_ids)


class FakeHandler:
    def __init__(self, text):
        self.content = [{"type": "text", "text": text}]
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.stop_reason = "end_turn"


class TestCompact(unittest.TestCase):
    """compact() with the model call stubbed out -- no network involved."""

    def make_agent(self, keep_last=2, fail=False):
        from lindwyrm.agent import Agent

        # compact_keep_tokens=1 leaves the protected zone driven purely by
        # the message count, which keeps these tests independent of how long
        # the fake messages happen to be.
        agent = Agent(make_config(compact_keep_last=keep_last,
                                  compact_keep_tokens=1, context_limit=1000))
        agent.messages = [
            user_text("a"), assistant_tool_use("t1"), tool_results("t1"),
            assistant_text("did a"),
            user_text("b"), assistant_text("did b"),
        ]

        self.calls = []

        def fake_call(messages, system, tools, *, cfg=None, **kw):
            self.calls.append({"system": system, "tools": tools, "cfg": cfg})
            if fail:
                raise RuntimeError("provider exploded")
            return FakeHandler("SUMMARY OF EARLIER WORK")

        agent._call_model = fake_call
        return agent

    def test_summary_call_sends_no_tools_and_disables_thinking(self):
        """Reasoning tokens on a summarization call are pure waste."""
        agent = self.make_agent()
        agent.compact()
        self.assertEqual(self.calls[0]["tools"], [])
        self.assertFalse(self.calls[0]["cfg"].thinking)

    def test_history_is_replaced_with_a_summary_pair(self):
        agent = self.make_agent()
        ok, note = agent.compact()
        self.assertTrue(ok, note)
        self.assertEqual(agent.messages[0]["role"], "user")
        self.assertIn("SUMMARY OF EARLIER WORK", agent.messages[0]["content"][0]["text"])
        self.assertEqual(agent.messages[1]["role"], "assistant")

    def test_roles_alternate_after_compaction(self):
        agent = self.make_agent()
        agent.compact()
        roles = [m["role"] for m in agent.messages]
        for a, b in zip(roles, roles[1:]):
            self.assertNotEqual(a, b, f"consecutive {a} messages: {roles}")

    def test_compaction_shortens_history(self):
        agent = self.make_agent()
        before = len(agent.messages)
        agent.compact()
        self.assertLess(len(agent.messages), before)

    def test_a_failed_summary_leaves_history_untouched(self):
        agent = self.make_agent(fail=True)
        original = list(agent.messages)
        ok, note = agent.compact()
        self.assertFalse(ok)
        self.assertIn("failed", note)
        self.assertEqual(agent.messages, original)

    def test_nothing_to_compact_is_reported_not_raised(self):
        agent = self.make_agent(keep_last=99)
        ok, note = agent.compact()
        self.assertFalse(ok)
        self.assertEqual(agent.messages[0], user_text("a"))


def long_turn(steps, request="port the parser to the new API", size=40):
    """One request followed by `steps` tool cycles and nothing else."""
    msgs = [user_text(request)]
    for i in range(steps):
        msgs.append(assistant_tool_use(f"t{i}"))
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "r" * size}]})
    return msgs


def assert_pairs_intact(case, messages):
    """Every tool_result answers a tool_use in the message just before it."""
    for i, m in enumerate(messages):
        if m["role"] != "user":
            continue
        ids = [b["tool_use_id"] for b in m["content"] if b.get("type") == "tool_result"]
        if not ids:
            continue
        case.assertGreater(i, 0)
        asked = {b["id"] for b in messages[i - 1]["content"]
                 if b.get("type") == "tool_use"}
        case.assertTrue(set(ids) <= asked, f"orphaned tool_result at {i}")


class TestMidturnCut(unittest.TestCase):
    """A single long turn has no boundary but index 0. It used to be
    impossible to compact, however far past the window it had grown."""

    def make_agent(self, messages, reply="NOTES", stop="end_turn", **kw):
        agent = Agent(make_config(compact_keep_last=2, compact_keep_tokens=1,
                                  context_limit=1000, **kw))
        agent.messages = messages
        self.sent = []

        def fake_call(messages, system, tools, *, cfg=None, **_):
            self.sent.append(messages)
            h = FakeHandler(reply)
            h.input_tokens, h.output_tokens, h.stop_reason = 5000, 300, stop
            return h

        agent._call_model = fake_call
        return agent

    def test_cut_lands_on_an_assistant_message(self):
        msgs = long_turn(5)
        for index in range(len(msgs) + 2):
            cut = midturn_cut_at_or_before(msgs, index)
            if cut:
                self.assertEqual(msgs[cut]["role"], "assistant")

    def test_the_request_alone_is_not_worth_cutting(self):
        self.assertEqual(midturn_cut_at_or_before(long_turn(1), 1), 0)

    def test_a_single_long_turn_is_compacted(self):
        agent = self.make_agent(long_turn(20))
        ok, note = agent.compact()
        self.assertTrue(ok, note)
        self.assertLess(len(agent.messages), 41)
        self.assertTrue(agent.messages[0]["content"][0]["text"].startswith(SUMMARY_PREFIX))

    def test_history_stays_valid_after_a_midturn_cut(self):
        agent = self.make_agent(long_turn(20))
        agent.compact()
        roles = [m["role"] for m in agent.messages]
        self.assertEqual(roles[0], "user")
        for a, b in zip(roles, roles[1:]):
            self.assertNotEqual(a, b, f"consecutive {a} messages: {roles}")
        assert_pairs_intact(self, agent.messages)

    def test_the_request_survives_word_for_word(self):
        agent = self.make_agent(long_turn(20, request="rename foo() to bar(), keep the shim"))
        agent.compact()
        head = agent.messages[0]["content"][0]["text"]
        self.assertTrue(head.endswith(
            REQUEST_MARKER + "rename foo() to bar(), keep the shim"))

    def test_the_request_survives_a_second_cut_of_the_same_turn(self):
        agent = self.make_agent(long_turn(20, request="rename foo() to bar()"))
        agent.compact()
        agent.messages += long_turn(20)[1:]
        ok, note = agent.compact()
        self.assertTrue(ok, note)
        head = agent.messages[0]["content"][0]["text"]
        self.assertEqual(head.count("rename foo() to bar()"), 1)
        self.assertTrue(head.endswith(REQUEST_MARKER + "rename foo() to bar()"))

    def test_an_enormous_request_is_left_to_the_summary(self):
        agent = self.make_agent(long_turn(20, request="x" * 4000))
        agent.compact()
        self.assertNotIn(REQUEST_MARKER, agent.messages[0]["content"][0]["text"])

    def test_the_session_keeps_its_title(self):
        agent = self.make_agent(long_turn(20, request="rename foo() to bar()"))
        agent.compact()
        self.assertEqual(_title_from(agent.messages), "rename foo() to bar()")

    def test_the_summary_request_never_follows_a_user_message(self):
        """`older` ends on tool results after a mid-turn cut; the question
        joins that message instead of making two user messages in a row."""
        agent = self.make_agent(long_turn(20))
        agent.compact()
        roles = [m["role"] for m in self.sent[0]]
        for a, b in zip(roles, roles[1:]):
            self.assertNotEqual(a, b)
        last = self.sent[0][-1]["content"]
        self.assertEqual(last[0]["type"], "tool_result")
        self.assertEqual(last[-1]["type"], "text")

    def test_the_original_history_is_not_edited_by_the_summary_call(self):
        msgs = long_turn(20)
        agent = self.make_agent(msgs)
        agent.compact()
        for m in msgs:
            if m["role"] == "user" and m["content"][0].get("type") == "tool_result":
                self.assertEqual(len(m["content"]), 1)

    def test_a_small_current_turn_is_still_kept_whole(self):
        """The turn boundary stays the preferred cut while the turn in
        progress is small next to the compaction trigger."""
        msgs = [user_text("a"), assistant_text("did a")] + long_turn(3, request="b")
        agent = self.make_agent(msgs, compact_max_tokens=0)
        agent.cfg.context_limit = 100_000
        agent.compact()
        self.assertEqual(agent.messages[2], user_text("b"))

    def test_a_huge_current_turn_is_cut_inside(self):
        msgs = [user_text("a"), assistant_text("did a")] + long_turn(20, request="b", size=400)
        agent = self.make_agent(msgs)
        agent.compact()
        self.assertEqual(agent.messages[1]["role"], "assistant")
        self.assertLess(len(agent.messages), 10)
        self.assertIn(REQUEST_MARKER + "b", agent.messages[0]["content"][0]["text"])

    def test_the_summary_call_is_counted_in_the_spend(self):
        """Usually the largest single request of the session; it used to be
        left out of the cost entirely."""
        agent = self.make_agent(long_turn(20))
        agent.compact()
        self.assertEqual(agent.total_fresh_input_tokens, 5000)
        self.assertEqual(agent.total_output_tokens, 300)

    def test_a_summary_cut_off_by_max_tokens_is_said_out_loud(self):
        agent = self.make_agent(long_turn(20), stop="max_tokens")
        ok, note = agent.compact()
        self.assertTrue(ok)
        self.assertIn("max_tokens", note)


class TestPendingEstimate(unittest.TestCase):
    """The API's count describes the last request. What was appended after
    it has to be added, or the trigger runs one step behind."""

    def test_results_appended_since_the_measurement_are_counted(self):
        agent = Agent(make_config())
        agent.messages = [user_text("go")]
        agent.last_input_tokens, agent._measured_upto = 1000, 1
        self.assertEqual(agent.context_tokens(), 1000)
        agent.messages += [assistant_tool_use(), {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 40_000}]}]
        self.assertGreater(agent.context_tokens(), 1000 + 9_000)

    def test_one_big_result_triggers_compaction_before_the_next_request(self):
        agent = Agent(make_config(context_limit=10_000, compact_max_tokens=0))
        agent.messages = [user_text("go"), assistant_text(), user_text("more")]
        agent.last_input_tokens, agent._measured_upto = 1000, 3
        self.assertFalse(agent.should_compact())
        agent.messages += [assistant_tool_use(), {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 40_000}]}]
        self.assertTrue(agent.should_compact())


class TestEstimateTokens(unittest.TestCase):
    def test_grows_with_content(self):
        small = estimate_tokens([user_text("hi")])
        large = estimate_tokens([user_text("hi" * 1000)])
        self.assertGreater(large, small)

    def test_counts_the_system_prompt(self):
        self.assertGreater(estimate_tokens([], "x" * 400), 0)

    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens([], ""), 0)


if __name__ == "__main__":
    unittest.main()
