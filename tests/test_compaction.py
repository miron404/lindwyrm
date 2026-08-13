"""History compaction: cutting in the wrong place is an API 400, so the
boundary logic is the part worth pinning down."""

import unittest

from lindwyrm.agent import estimate_tokens, find_cut_index, is_turn_boundary


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


class TestCompact(unittest.TestCase):
    """compact() with the model call stubbed out -- no network involved."""

    def make_agent(self, keep_last=2, fail=False):
        import dataclasses

        from lindwyrm.agent import Agent

        @dataclasses.dataclass
        class Cfg:
            compact_keep_last: int
            context_limit: int = 1000
            auto_compact: bool = True
            compact_threshold: float = 0.75
            # 1 token: the protected zone is then driven purely by
            # compact_keep_last, which keeps these tests deterministic
            # regardless of how long the fake messages happen to be.
            compact_keep_tokens: int = 1
            offload: bool = False
            offload_threshold_tokens: int = 1000
            offload_eager_tokens: int = 8000
            format: str = "anthropic"
            thinking: bool = True
            audit_log: object = None

        agent = Agent(Cfg(compact_keep_last=keep_last))
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
