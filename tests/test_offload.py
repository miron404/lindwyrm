"""Offloading bulky tool results to disk.

The property that matters: it is reversible. A summary throws information
away; an offloaded result is still on disk and can be fetched back, and the
stub says plainly that it is a point-in-time snapshot rather than the file's
current contents.
"""

import dataclasses
import tempfile
import time
import pathlib
import unittest
from pathlib import Path

from lindwyrm import offload
from lindwyrm.agent import Agent, protected_index
from lindwyrm.offload import OffloadStore, estimate_text_tokens
from lindwyrm.sandbox import SandboxError
from lindwyrm.tools import tool_read_offloaded


def user_text(text="hi"):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_text(text="ok"):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def tool_result(tool_id, body):
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": body}]}


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = OffloadStore(root=Path(self._tmp.name) / "session")
        offload.set_store(self.store)

    def tearDown(self):
        offload.set_store(None)
        self._tmp.cleanup()


class TestStore(StoreTestCase):
    def test_round_trip(self):
        entry = self.store.put("hello\nworld\n", "read_file a.py")
        self.assertIsNotNone(entry)
        self.assertEqual(self.store.get(entry.ref), "hello\nworld\n")

    def test_refs_are_unique(self):
        a = self.store.put("one", "x")
        b = self.store.put("two", "y")
        self.assertNotEqual(a.ref, b.ref)
        self.assertEqual(self.store.get(a.ref), "one")
        self.assertEqual(self.store.get(b.ref), "two")

    def test_line_range(self):
        entry = self.store.put("1\n2\n3\n4\n5\n", "x")
        self.assertEqual(self.store.get(entry.ref, 2, 4), "2\n3\n4")

    def test_unknown_ref_raises_with_a_useful_message(self):
        self.store.put("x", "y")
        with self.assertRaises(KeyError) as ctx:
            self.store.get("off_9999")
        self.assertIn("off_0001", str(ctx.exception))

    def test_content_is_stored_as_plain_text(self):
        """Kept greppable from outside -- useful when debugging a session."""
        entry = self.store.put("needle here", "x")
        self.assertIn("needle here", entry.path.read_text())


class TestStub(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.text = "\n".join(f"line {i}" for i in range(200))
        self.entry = self.store.put(self.text, "read_file src/parser.py")
        self.stub = self.store.stub(self.entry, self.text)

    def test_stub_is_far_smaller_than_the_content(self):
        self.assertLess(len(self.stub), len(self.text) / 4)

    def test_stub_names_the_source_and_the_ref(self):
        self.assertIn("read_file src/parser.py", self.stub)
        self.assertIn(self.entry.ref, self.stub)

    def test_stub_previews_the_beginning(self):
        self.assertIn("line 0", self.stub)

    def test_stub_says_it_is_a_snapshot(self):
        """Without this the model would assume it reflects the file now, and
        reason against a version that may since have changed or been deleted."""
        self.assertIn("snapshot", self.stub.lower())
        self.assertIn("read_file", self.stub)


class TestReadOffloadedTool(StoreTestCase):
    def test_tool_returns_the_snapshot(self):
        entry = self.store.put("secret contents", "bash ls")
        self.assertEqual(tool_read_offloaded(None, entry.ref), "secret contents")

    def test_tool_reports_a_bad_ref_as_a_tool_error(self):
        with self.assertRaises(SandboxError):
            tool_read_offloaded(None, "off_nope")

    def test_snapshot_survives_the_original_file_changing(self):
        """The whole reason for copying rather than re-reading."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.py"
            f.write_text("ORIGINAL")
            entry = self.store.put(f.read_text(), f"read_file {f}")
            f.write_text("REWRITTEN")
            self.assertEqual(tool_read_offloaded(None, entry.ref), "ORIGINAL")

    def test_snapshot_survives_the_original_file_being_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.py"
            f.write_text("ORIGINAL")
            entry = self.store.put(f.read_text(), f"read_file {f}")
            f.unlink()
            self.assertEqual(tool_read_offloaded(None, entry.ref), "ORIGINAL")


@dataclasses.dataclass
class Cfg:
    context_limit: int = 100_000
    compact_keep_last: int = 2
    compact_keep_tokens: int = 1
    offload: bool = True
    offload_threshold_tokens: int = 10
    offload_eager_tokens: int = 100_000
    auto_compact: bool = True
    compact_threshold: float = 0.75
    format: str = "anthropic"
    thinking: bool = True
    audit_log: object = None
    project_root: object = pathlib.Path("/nonexistent-project")
    user_context_dir: object = pathlib.Path("/nonexistent-userdir")


class TestMicrocompact(StoreTestCase):
    def make_agent(self, **kw):
        agent = Agent(Cfg(**kw))
        big = "x" * 4000
        agent.messages = [
            user_text("do the thing"),
            assistant_text("reading"),
            tool_result("t1", big),
            assistant_text("read it"),
            user_text("now this"),
            assistant_text("done"),
        ]
        agent._tool_labels["t1"] = "read_file big.py"
        return agent

    def test_big_old_result_is_offloaded(self):
        agent = self.make_agent()
        moved, freed = agent.microcompact()
        self.assertEqual(moved, 1)
        self.assertGreater(freed, 0)
        body = agent.messages[2]["content"][0]["content"]
        self.assertTrue(body.startswith("[offloaded:"))
        self.assertIn("read_file big.py", body)

    def test_offloaded_content_is_recoverable(self):
        agent = self.make_agent()
        agent.microcompact()
        body = agent.messages[2]["content"][0]["content"]
        ref = body.split("ref ")[1].split("]")[0]
        self.assertEqual(len(self.store.get(ref)), 4000)

    def test_recent_results_are_protected(self):
        """The protected zone is what keeps the current task's context intact."""
        agent = self.make_agent()
        agent.messages.append(tool_result("t2", "y" * 4000))
        agent._tool_labels["t2"] = "bash pytest"
        agent.microcompact()
        self.assertEqual(agent.messages[-1]["content"][0]["content"], "y" * 4000)

    def test_small_results_are_left_alone(self):
        agent = self.make_agent(offload_threshold_tokens=100_000)
        moved, _ = agent.microcompact()
        self.assertEqual(moved, 0)

    def test_disabled_by_config(self):
        agent = self.make_agent(offload=False)
        self.assertEqual(agent.microcompact(), (0, 0))

    def test_running_twice_does_not_double_offload(self):
        agent = self.make_agent()
        agent.microcompact()
        moved, _ = agent.microcompact()
        self.assertEqual(moved, 0)

    def test_it_shrinks_the_context_estimate(self):
        agent = self.make_agent()
        before = agent.context_tokens()
        agent.microcompact()
        self.assertLess(agent.context_tokens(), before)


class TestProtectedZone(unittest.TestCase):
    def test_zone_is_measured_in_tokens_not_messages(self):
        """Same message count, very different sizes: the zone must stop after
        the huge one and keep going through the small ones."""
        big = [user_text("a"), assistant_text("b"), user_text("x" * 40_000)]
        small = [user_text("a"), assistant_text("b"), user_text("c")]
        self.assertEqual(
            protected_index(big, keep_tokens=1_000, keep_last=1), 2)
        self.assertEqual(
            protected_index(small, keep_tokens=1_000, keep_last=1), 0)

    def test_message_count_is_a_floor(self):
        msgs = [user_text("a"), assistant_text("b"), user_text("c"),
                assistant_text("d")]
        index = protected_index(msgs, keep_tokens=1, keep_last=3)
        self.assertLessEqual(index, len(msgs) - 3)

    def test_zone_is_capped_against_a_small_context_window(self):
        """A flat 8k zone would protect everything in an 8k window and
        compaction would never fire at all."""
        agent = Agent(Cfg(context_limit=4_000, compact_keep_tokens=8_000))
        self.assertLessEqual(agent.keep_tokens(), 1_000)

    def test_configured_zone_is_used_when_it_fits(self):
        agent = Agent(Cfg(context_limit=200_000, compact_keep_tokens=8_000))
        self.assertEqual(agent.keep_tokens(), 8_000)


class TestSweep(unittest.TestCase):
    def test_old_session_directories_are_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            ancient = time.time() - 30 * 86400
            import os
            os.utime(old, (ancient, ancient))

            removed = offload.sweep_old_sessions(root, max_age_days=7)
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_missing_root_is_not_an_error(self):
        self.assertEqual(
            offload.sweep_old_sessions(Path("/nonexistent/lindwyrm"), 7), 0)


class TestEstimate(unittest.TestCase):
    def test_scales_with_length(self):
        self.assertGreater(estimate_text_tokens("x" * 400),
                           estimate_text_tokens("x" * 4))


if __name__ == "__main__":
    unittest.main()
