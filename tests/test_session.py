"""Saving and resuming conversations."""

import dataclasses
import json
import pathlib
import tempfile
import time
import unittest
from pathlib import Path

from lindwyrm import offload, session
from lindwyrm.agent import Agent
from lindwyrm.offload import OffloadStore


def user_text(text="hi"):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_thinking(text="reasoning"):
    return {"role": "assistant", "content": [
        {"type": "thinking", "thinking": text, "signature": "sig"},
        {"type": "text", "text": "answer"}]}


@dataclasses.dataclass
class Cfg:
    context_limit: int = 100_000
    compact_keep_last: int = 2
    compact_keep_tokens: int = 1
    offload: bool = True
    offload_threshold_tokens: int = 1000
    offload_eager_tokens: int = 8000
    auto_compact: bool = True
    compact_threshold: float = 0.75
    format: str = "anthropic"
    thinking: bool = True
    audit_log: object = None
    project_root: object = pathlib.Path("/nonexistent-project")
    user_context_dir: object = pathlib.Path("/nonexistent-userdir")
    price_input: float | None = None
    price_output: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestIds(unittest.TestCase):
    def test_ids_are_unique(self):
        self.assertNotEqual(session.new_session_id(), session.new_session_id())

    def test_ids_sort_chronologically(self):
        """Listing relies on this instead of stat-ing every file."""
        first = session.new_session_id()
        time.sleep(1.01)
        self.assertLess(first, session.new_session_id())


class TestRoundTrip(SessionTestCase):
    def test_save_then_load(self):
        state = {"messages": [user_text("hello")], "project_root": "/p"}
        session.save_session("s1", state, root=self.root)
        loaded = session.load_session("s1", root=self.root)
        self.assertEqual(loaded["messages"], state["messages"])

    def test_thinking_blocks_survive(self):
        """A history saved without them resumes straight into a 400 from
        DeepSeek, which requires them echoed back."""
        state = {"messages": [user_text(), assistant_thinking("deep thought")]}
        session.save_session("s1", state, root=self.root)
        loaded = session.load_session("s1", root=self.root)
        blocks = loaded["messages"][1]["content"]
        self.assertEqual(blocks[0]["type"], "thinking")
        self.assertEqual(blocks[0]["thinking"], "deep thought")
        self.assertEqual(blocks[0]["signature"], "sig")

    def test_unicode_survives(self):
        session.save_session("s1", {"messages": [user_text("привет 🐉")]},
                             root=self.root)
        loaded = session.load_session("s1", root=self.root)
        self.assertIn("привет 🐉", json.dumps(loaded, ensure_ascii=False))

    def test_loading_a_missing_session_returns_none(self):
        self.assertIsNone(session.load_session("nope", root=self.root))

    def test_a_corrupt_file_returns_none_rather_than_raising(self):
        (self.root / "bad.json").write_text("{not json")
        self.assertIsNone(session.load_session("bad", root=self.root))

    def test_written_atomically(self):
        """A crash mid-write must not leave a truncated file where a session
        used to be."""
        session.save_session("s1", {"messages": [user_text("first")]},
                             root=self.root)
        session.save_session("s1", {"messages": [user_text("second")]},
                             root=self.root)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertIn("second", json.dumps(
            session.load_session("s1", root=self.root)))

    def test_files_are_not_world_readable(self):
        """They hold source code and command output."""
        path = session.save_session("s1", {"messages": []}, root=self.root)
        self.assertEqual(path.stat().st_mode & 0o077, 0)


class TestListing(SessionTestCase):
    def make(self, sid, project, text):
        session.save_session(sid, {"messages": [user_text(text)],
                                   "project_root": project}, root=self.root)

    def test_sessions_are_scoped_to_their_project(self):
        """--continue in one checkout must never resume another's work."""
        self.make("a", "/project/one", "one")
        self.make("b", "/project/two", "two")
        listed = session.list_sessions("/project/one", root=self.root)
        self.assertEqual([s["id"] for s in listed], ["a"])

    def test_newest_first(self):
        self.make("20260101-000000-aaaa", "/p", "old")
        self.make("20260202-000000-bbbb", "/p", "new")
        listed = session.list_sessions("/p", root=self.root)
        self.assertEqual(listed[0]["id"], "20260202-000000-bbbb")

    def test_limit_is_respected(self):
        for i in range(5):
            self.make(f"s{i}", "/p", f"m{i}")
        self.assertEqual(len(session.list_sessions("/p", limit=2, root=self.root)), 2)

    def test_title_comes_from_the_first_real_prompt(self):
        self.make("a", "/p", "fix the parser bug")
        self.assertIn("fix the parser", session.list_sessions("/p", root=self.root)[0]["title"])

    def test_a_compaction_summary_is_not_used_as_a_title(self):
        session.save_session("a", {"messages": [
            user_text("[Summary of earlier conversation, compacted...]"),
            user_text("the actual question")], "project_root": "/p"},
            root=self.root)
        title = session.list_sessions("/p", root=self.root)[0]["title"]
        self.assertIn("actual question", title)

    def test_a_corrupt_file_does_not_break_the_listing(self):
        self.make("good", "/p", "fine")
        (self.root / "bad.json").write_text("{{{")
        self.assertEqual(len(session.list_sessions("/p", root=self.root)), 1)

    def test_latest_for_a_project(self):
        self.make("20260101-000000-aaaa", "/p", "old")
        self.make("20260303-000000-cccc", "/p", "new")
        self.assertEqual(session.latest_session_id("/p", root=self.root),
                         "20260303-000000-cccc")

    def test_latest_is_none_when_the_project_has_none(self):
        self.assertIsNone(session.latest_session_id("/empty", root=self.root))


class TestSweep(SessionTestCase):
    def test_old_sessions_are_removed(self):
        session.save_session("old", {"messages": []}, root=self.root)
        session.save_session("new", {"messages": []}, root=self.root)
        ancient = time.time() - 90 * 86400
        import os
        os.utime(self.root / "old.json", (ancient, ancient))

        removed = session.sweep_sessions(30, root=self.root)
        self.assertEqual(removed, 1)
        self.assertFalse((self.root / "old.json").exists())
        self.assertTrue((self.root / "new.json").exists())

    def test_missing_directory_is_not_an_error(self):
        self.assertEqual(session.sweep_sessions(30, Path("/nonexistent/x")), 0)


class TestAgentSnapshot(SessionTestCase):
    def setUp(self):
        super().setUp()
        self.store = OffloadStore(root=self.root / "off")
        offload.set_store(self.store)

    def tearDown(self):
        offload.set_store(None)
        super().tearDown()

    def test_history_and_counters_round_trip(self):
        agent = Agent(Cfg())
        agent.messages = [user_text("q"), assistant_thinking()]
        agent.last_input_tokens = 1234
        agent.total_output_tokens = 55
        agent.total_cache_read_tokens = 900
        agent.total_fresh_input_tokens = 300
        agent._tool_labels["t1"] = "read_file a.py"

        restored = Agent(Cfg())
        restored.restore(agent.snapshot())

        self.assertEqual(restored.messages, agent.messages)
        self.assertEqual(restored.last_input_tokens, 1234)
        self.assertEqual(restored.total_output_tokens, 55)
        self.assertEqual(restored.total_cache_read_tokens, 900)
        self.assertEqual(restored._tool_labels["t1"], "read_file a.py")

    def test_offloaded_results_are_still_retrievable(self):
        """Otherwise every [offloaded: ...] stub in a resumed history points
        at a ref that no longer resolves."""
        agent = Agent(Cfg())
        entry = self.store.put("the full content", "read_file big.py")
        state = agent.snapshot()

        offload.set_store(OffloadStore(root=self.root / "off"))
        Agent(Cfg()).restore(state)
        self.assertEqual(offload.get_store().get(entry.ref), "the full content")

    def test_offloaded_entries_whose_files_vanished_are_dropped(self):
        agent = Agent(Cfg())
        entry = self.store.put("gone soon", "bash ls")
        state = agent.snapshot()
        entry.path.unlink()

        fresh = OffloadStore(root=self.root / "off")
        offload.set_store(fresh)
        Agent(Cfg()).restore(state)
        self.assertEqual(len(fresh), 0)

    def test_new_refs_do_not_collide_with_restored_ones(self):
        agent = Agent(Cfg())
        self.store.put("one", "a")
        self.store.put("two", "b")
        state = agent.snapshot()

        fresh = OffloadStore(root=self.root / "off")
        offload.set_store(fresh)
        Agent(Cfg()).restore(state)
        new_entry = fresh.put("three", "c")
        self.assertEqual(fresh.get(new_entry.ref), "three")
        self.assertEqual(len(fresh), 3)

    def test_restoring_an_empty_state_is_safe(self):
        agent = Agent(Cfg())
        agent.restore({})
        self.assertEqual(agent.messages, [])


class TestDescribeAge(unittest.TestCase):
    def test_recent(self):
        self.assertEqual(session.describe_age(time.time()), "just now")

    def test_hours(self):
        self.assertIn("h ago", session.describe_age(time.time() - 3 * 3600))

    def test_days(self):
        self.assertIn("d ago", session.describe_age(time.time() - 5 * 86400))


if __name__ == "__main__":
    unittest.main()
