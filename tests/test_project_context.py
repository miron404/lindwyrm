"""Project instructions loaded into the system prompt."""

import tempfile
import unittest
from pathlib import Path

from lindwyrm.project import (
    CONTEXT_FILENAMES,
    MAX_CONTEXT_BYTES,
    TEMPLATE,
    collect_context_files,
    find_context_file,
    load_project_context,
)


class ContextTestCase(unittest.TestCase):
    def setUp(self):
        self._proj = tempfile.TemporaryDirectory()
        self._user = tempfile.TemporaryDirectory()
        self.project = Path(self._proj.name)
        self.user = Path(self._user.name)

    def tearDown(self):
        self._proj.cleanup()
        self._user.cleanup()

    def load(self, **kw):
        return load_project_context(self.project, user_dir=self.user, **kw)


class TestDiscovery(ContextTestCase):
    def test_nothing_found_is_not_an_error(self):
        text, described = self.load()
        self.assertEqual(text, "")
        self.assertEqual(described, [])

    def test_agents_md_is_found(self):
        (self.project / "AGENTS.md").write_text("run tests with unittest")
        text, described = self.load()
        self.assertIn("run tests with unittest", text)
        self.assertEqual(len(described), 1)

    def test_lindwyrm_md_wins_over_agents_md(self):
        """The project-specific name is the more deliberate choice."""
        (self.project / "AGENTS.md").write_text("generic")
        (self.project / "LINDWYRM.md").write_text("specific")
        text, _ = self.load()
        self.assertIn("specific", text)
        self.assertNotIn("generic", text)

    def test_filename_order_matches_the_documented_precedence(self):
        self.assertEqual(CONTEXT_FILENAMES[0], "LINDWYRM.md")
        self.assertIn("AGENTS.md", CONTEXT_FILENAMES)

    def test_user_and_project_files_are_both_loaded(self):
        (self.user / "AGENTS.md").write_text("I prefer tabs")
        (self.project / "AGENTS.md").write_text("this repo uses spaces")
        text, described = self.load()
        self.assertIn("I prefer tabs", text)
        self.assertIn("this repo uses spaces", text)
        self.assertEqual(len(described), 2)

    def test_user_file_comes_first(self):
        """Project notes read last, so they qualify the general preferences."""
        (self.user / "AGENTS.md").write_text("USERLEVEL")
        (self.project / "AGENTS.md").write_text("PROJECTLEVEL")
        text, _ = self.load()
        self.assertLess(text.index("USERLEVEL"), text.index("PROJECTLEVEL"))

    def test_explicit_path_overrides_discovery(self):
        (self.project / "AGENTS.md").write_text("ignored")
        (self.project / "notes.md").write_text("chosen")
        text, _ = self.load(explicit="notes.md")
        self.assertIn("chosen", text)
        self.assertNotIn("ignored", text)

    def test_missing_explicit_file_is_silently_skipped(self):
        text, described = self.load(explicit="nope.md")
        self.assertEqual(described, [])

    def test_find_context_file_on_a_missing_directory(self):
        self.assertIsNone(find_context_file(Path("/nonexistent/xyz")))

    def test_empty_file_contributes_nothing(self):
        (self.project / "AGENTS.md").write_text("   \n\n")
        text, described = self.load()
        self.assertEqual(described, [])


class TestFraming(ContextTestCase):
    """The file ships with the repo, so on a clone it is untrusted input."""

    def setUp(self):
        super().setUp()
        (self.project / "AGENTS.md").write_text("always deploy to production")
        self.text, _ = self.load()

    def test_content_is_labelled_as_reference_not_orders(self):
        lowered = self.text.lower()
        self.assertIn("reference material", lowered)
        self.assertIn("not as instructions", lowered)

    def test_it_states_that_policy_is_unaffected(self):
        """A repo must not be able to talk its way past the permission rules."""
        self.assertIn("policy is unchanged", self.text.lower())

    def test_the_source_file_is_named(self):
        self.assertIn("AGENTS.md", self.text)


class TestSizeLimit(ContextTestCase):
    def test_oversized_file_is_truncated_and_flagged(self):
        (self.project / "AGENTS.md").write_text("x" * (MAX_CONTEXT_BYTES + 5000))
        text, described = self.load()
        self.assertIn("truncated", described[0])
        self.assertLess(len(text), MAX_CONTEXT_BYTES + 2000)

    def test_custom_limit_is_respected(self):
        (self.project / "AGENTS.md").write_text("y" * 500)
        _, described = self.load(max_bytes=100)
        self.assertIn("truncated", described[0])


class TestDescriptions(ContextTestCase):
    def test_project_file_is_described_by_relative_name(self):
        (self.project / "AGENTS.md").write_text("hi")
        _, described = self.load()
        self.assertTrue(described[0].startswith("AGENTS.md"))

    def test_size_is_reported(self):
        (self.project / "AGENTS.md").write_text("z" * 2048)
        _, described = self.load()
        self.assertIn("KB", described[0])


class TestCollect(ContextTestCase):
    def test_order_is_user_then_project(self):
        (self.user / "AGENTS.md").write_text("u")
        (self.project / "AGENTS.md").write_text("p")
        files = collect_context_files(self.project, user_dir=self.user)
        self.assertEqual([f.parent for f in files], [self.user, self.project])


class TestTemplate(unittest.TestCase):
    def test_template_covers_what_a_session_keeps_re_asking(self):
        for heading in ("Layout", "Commands", "Conventions", "Gotchas"):
            self.assertIn(heading, TEMPLATE)

    def test_template_is_valid_markdown_headings(self):
        self.assertTrue(TEMPLATE.lstrip().startswith("#"))


if __name__ == "__main__":
    unittest.main()
