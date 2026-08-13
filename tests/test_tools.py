"""File tools: editing, searching, and the guards around them."""

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from lindwyrm.config import Policy
from lindwyrm.sandbox import SandboxError
from lindwyrm.tools import (
    MAX_DIFF_LINES,
    _looks_binary,
    _match_lines,
    _unified_diff,
    tool_delete_file,
    tool_edit_file,
    tool_glob,
    tool_write_file,
    tool_grep,
    tool_list_dir,
    tool_read_file,
)


@dataclass
class FakeConfig:
    project_root: Path
    policy: Policy


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.cfg = FakeConfig(self.root, Policy(read="allow", write="allow"))

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p


class TestMatchLines(unittest.TestCase):
    def test_reports_1_based_line_numbers(self):
        text = "alpha\nbeta\nalpha\n"
        self.assertEqual(_match_lines(text, "alpha"), [1, 3])

    def test_no_matches(self):
        self.assertEqual(_match_lines("abc", "zzz"), [])

    def test_multiline_needle(self):
        self.assertEqual(_match_lines("a\nb\nc\n", "b\nc"), [2])

    def test_empty_needle_terminates(self):
        """A zero-length match must not loop forever."""
        self.assertIsInstance(_match_lines("abc", ""), list)


class TestEditFile(ToolTestCase):
    def test_unique_match_is_replaced(self):
        self.write("a.py", "x = 1\ny = 2\n")
        tool_edit_file(self.cfg, "a.py", "y = 2", "y = 3")
        self.assertEqual((self.root / "a.py").read_text(), "x = 1\ny = 3\n")

    def test_missing_text_is_an_error(self):
        self.write("a.py", "x = 1\n")
        with self.assertRaises(SandboxError) as ctx:
            tool_edit_file(self.cfg, "a.py", "nope", "y")
        self.assertIn("not found", str(ctx.exception))

    def test_ambiguous_match_reports_the_line_numbers(self):
        """The old message just said 'must be unique', leaving the model to
        guess where the duplicates were."""
        self.write("a.py", "val = 0\nother = 1\nval = 0\n")
        with self.assertRaises(SandboxError) as ctx:
            tool_edit_file(self.cfg, "a.py", "val = 0", "val = 9")
        msg = str(ctx.exception)
        self.assertIn("appears 2 times", msg)
        self.assertIn("line 1", msg)
        self.assertIn("line 3", msg)
        self.assertIn("replace_all", msg)

    def test_replace_all_changes_every_occurrence(self):
        self.write("a.py", "v = 0\nx = 1\nv = 0\n")
        out = tool_edit_file(self.cfg, "a.py", "v = 0", "v = 9", replace_all=True)
        self.assertEqual((self.root / "a.py").read_text(), "v = 9\nx = 1\nv = 9\n")
        self.assertIn("2 occurrences", out)

    def test_replace_all_still_errors_when_nothing_matches(self):
        self.write("a.py", "x\n")
        with self.assertRaises(SandboxError):
            tool_edit_file(self.cfg, "a.py", "zzz", "y", replace_all=True)

    def test_editing_a_directory_is_refused(self):
        (self.root / "sub").mkdir()
        with self.assertRaises(SandboxError):
            tool_edit_file(self.cfg, "sub", "a", "b")


class TestUnifiedDiff(unittest.TestCase):
    """The confirmation prompt is where you decide yes or no in a second or
    two, so the diff has to show what changed, not restate the whole block."""

    def setUp(self):
        self.old = "\n".join(f"line {i}" for i in range(1, 21))
        self.new = self.old.replace("line 8", "line EIGHT")

    def test_only_the_changed_hunk_is_shown(self):
        out = _unified_diff(self.old, self.new, "a.py")
        self.assertIn("-line 8", out)
        self.assertIn("+line EIGHT", out)
        # Untouched lines far from the change stay out of it.
        self.assertNotIn("line 1\n", out)
        self.assertNotIn("line 20", out)

    def test_it_is_far_shorter_than_listing_both_versions(self):
        out = _unified_diff(self.old, self.new, "a.py")
        self.assertLess(len(out.splitlines()), 20)

    def test_context_lines_surround_the_change(self):
        out = _unified_diff(self.old, self.new, "a.py")
        self.assertIn(" line 7", out)
        self.assertIn(" line 9", out)

    def test_counts_are_summarised(self):
        self.assertIn("+1 -1", _unified_diff(self.old, self.new, "a.py"))

    def test_hunk_headers_carry_line_numbers(self):
        self.assertIn("@@", _unified_diff(self.old, self.new, "a.py"))

    def test_identical_content_says_so(self):
        self.assertIn("no textual change", _unified_diff("x", "x", "a.py"))

    def test_a_whole_file_rewrite_is_truncated(self):
        old = "\n".join(f"old {i}" for i in range(200))
        new = "\n".join(f"new {i}" for i in range(200))
        out = _unified_diff(old, new, "big.py")
        self.assertLessEqual(len(out.splitlines()), MAX_DIFF_LINES + 3)
        self.assertIn("more diff lines", out)

    def test_creating_content_from_nothing(self):
        out = _unified_diff("", "hello\n", "new.py")
        self.assertIn("+hello", out)


class TestWritePreview(ToolTestCase):
    def test_overwriting_an_existing_file_previews_a_diff(self):
        """The new content alone doesn't show what is about to be lost."""
        self.write("a.py", "keep = 1\ndrop = 2\n")
        seen = {}

        import lindwyrm.tools as tools
        original = tools.authorize

        def spy(cfg, op, path, summary, preview=None):
            seen["preview"] = preview
            return original(cfg, op, path, summary, preview=preview)

        tools.authorize = spy
        try:
            tool_write_file(self.cfg, "a.py", "keep = 1\nadded = 3\n")
        finally:
            tools.authorize = original

        self.assertIn("-drop = 2", seen["preview"])
        self.assertIn("+added = 3", seen["preview"])

    def test_creating_a_new_file_previews_its_content(self):
        seen = {}
        import lindwyrm.tools as tools
        original = tools.authorize

        def spy(cfg, op, path, summary, preview=None):
            seen["preview"] = preview
            return original(cfg, op, path, summary, preview=preview)

        tools.authorize = spy
        try:
            tool_write_file(self.cfg, "brand_new.py", "print('hi')\n")
        finally:
            tools.authorize = original
        self.assertIn("print('hi')", seen["preview"])


class TestGrep(ToolTestCase):
    def test_finds_matches_with_line_numbers(self):
        self.write("a.py", "import os\nimport sys\n")
        out = tool_grep(self.cfg, "import sys")
        self.assertIn("a.py:2:", out)

    def test_regex_search(self):
        self.write("a.py", "cost = a[0]\n")
        self.assertIn("a.py:1:", tool_grep(self.cfg, r"cost = \w\[0\]"))

    def test_invalid_regex_falls_back_to_a_literal_search(self):
        """`a[0` is an unterminated character class -- re.compile raises, and
        the pattern is searched for verbatim instead of erroring out."""
        self.write("a.py", "cost = a[0]\n")
        self.assertIn("a.py:1:", tool_grep(self.cfg, "a[0"))

    def test_skips_vendored_directories(self):
        self.write("node_modules/dep/index.js", "NEEDLE\n")
        self.write("src/app.js", "NEEDLE\n")
        out = tool_grep(self.cfg, "NEEDLE")
        self.assertIn("src/app.js", out)
        self.assertNotIn("node_modules", out)

    def test_skips_binary_files(self):
        (self.root / "blob.bin").write_bytes(b"NEEDLE\x00\x01\x02")
        self.write("ok.txt", "NEEDLE\n")
        out = tool_grep(self.cfg, "NEEDLE")
        self.assertIn("ok.txt", out)
        self.assertNotIn("blob.bin", out)

    def test_no_matches_message(self):
        self.write("a.py", "nothing here\n")
        self.assertEqual(tool_grep(self.cfg, "ZZZ"), "(no matches)")

    def test_path_may_be_a_single_file(self):
        """This used to answer '(no matches)' for a file full of matches --
        rglob on a file yields nothing -- and the model believed it."""
        self.write("a.py", "NEEDLE here\n")
        self.write("b.py", "NEEDLE there\n")
        out = tool_grep(self.cfg, "NEEDLE", path="a.py")
        self.assertIn("a.py:1:", out)
        self.assertNotIn("b.py", out)

    def test_missing_needle_in_a_single_file_still_reports_no_matches(self):
        self.write("a.py", "nothing\n")
        self.assertEqual(tool_grep(self.cfg, "ZZZ", path="a.py"), "(no matches)")

    def test_glob_filter(self):
        self.write("a.py", "NEEDLE\n")
        self.write("b.txt", "NEEDLE\n")
        out = tool_grep(self.cfg, "NEEDLE", glob="*.py")
        self.assertIn("a.py", out)
        self.assertNotIn("b.txt", out)


class TestGlob(ToolTestCase):
    def test_skips_vendored_directories(self):
        self.write("node_modules/x/a.js", "")
        self.write(".venv/lib/b.py", "")
        self.write("src/c.py", "")
        out = tool_glob(self.cfg, "**/*.py")
        self.assertIn("src/c.py", out)
        self.assertNotIn(".venv", out)

    def test_no_matches_message(self):
        self.assertEqual(tool_glob(self.cfg, "**/*.zzz"), "(no matches)")

    def test_a_file_as_base_is_an_explicit_error(self):
        """Silently returning nothing would read as 'no such files exist'."""
        self.write("a.py", "x")
        with self.assertRaises(SandboxError) as ctx:
            tool_glob(self.cfg, "*", path="a.py")
        self.assertIn("not a directory", str(ctx.exception))


class TestBinarySniff(ToolTestCase):
    def test_text_file_is_not_binary(self):
        p = self.write("a.txt", "hello\n")
        self.assertFalse(_looks_binary(p))

    def test_nul_byte_means_binary(self):
        p = self.root / "b.bin"
        p.write_bytes(b"abc\x00def")
        self.assertTrue(_looks_binary(p))


class TestListDir(ToolTestCase):
    def test_dotfiles_and_dot_directories_are_listed(self):
        """They used to be skipped silently, which hid .github/, .gitignore
        and every config file -- and glob never returns directories, so a
        hidden directory was unreachable by any read tool."""
        self.write(".gitignore", "*.pyc")
        self.write(".github/workflows/ci.yml", "name: ci")
        self.write("visible.py", "x")
        out = tool_list_dir(self.cfg)
        self.assertIn(".gitignore", out)
        self.assertIn(".github/", out)
        self.assertIn("visible.py", out)

    def test_directories_sort_before_files(self):
        self.write("a_file.txt", "x")
        (self.root / "z_dir").mkdir()
        out = tool_list_dir(self.cfg).splitlines()
        self.assertLess(out.index("z_dir/"), out.index("a_file.txt"))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlinks_are_marked_with_their_target(self):
        self.write("real.txt", "x")
        (self.root / "link.txt").symlink_to("real.txt")
        self.assertIn("link.txt@ -> real.txt", tool_list_dir(self.cfg))

    def test_missing_directory_says_so(self):
        with self.assertRaises(SandboxError) as ctx:
            tool_list_dir(self.cfg, "nope")
        self.assertIn("Does not exist", str(ctx.exception))

    def test_empty_directory(self):
        (self.root / "empty").mkdir()
        self.assertEqual(tool_list_dir(self.cfg, "empty"), "(empty directory)")


class TestDeleteFile(ToolTestCase):
    def setUp(self):
        super().setUp()
        # delete defaults to "confirm"; leaving it there makes the tool block
        # on input() waiting for a confirmation nobody is present to give.
        self.cfg.policy.delete = "allow"

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_a_broken_symlink_can_be_deleted(self):
        """exists() follows the link, so a dangling one could never be
        removed -- it reported that it did not exist."""
        (self.root / "dangling").symlink_to(self.root / "gone")
        tool_delete_file(self.cfg, "dangling")
        self.assertFalse((self.root / "dangling").is_symlink())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_deleting_a_symlink_leaves_its_target(self):
        self.write("real.txt", "keep me")
        (self.root / "link.txt").symlink_to(self.root / "real.txt")
        tool_delete_file(self.cfg, "link.txt")
        self.assertEqual((self.root / "real.txt").read_text(), "keep me")

    def test_directories_are_refused(self):
        (self.root / "sub").mkdir()
        with self.assertRaises(SandboxError):
            tool_delete_file(self.cfg, "sub")


class TestReadFile(ToolTestCase):
    def test_line_numbers_are_prefixed(self):
        self.write("a.py", "one\ntwo\n")
        out = tool_read_file(self.cfg, "a.py")
        self.assertIn("1\tone", out)
        self.assertIn("2\ttwo", out)

    def test_line_range(self):
        self.write("a.py", "1\n2\n3\n4\n")
        out = tool_read_file(self.cfg, "a.py", start_line=2, end_line=3)
        self.assertNotIn("\t1", out)
        self.assertIn("\t2", out)
        self.assertIn("\t3", out)

    def test_out_of_range_start_is_an_error_not_an_empty_file(self):
        """It used to answer '(empty file)' for a perfectly full file."""
        self.write("a.py", "one\ntwo\n")
        with self.assertRaises(SandboxError) as ctx:
            tool_read_file(self.cfg, "a.py", start_line=99)
        msg = str(ctx.exception)
        self.assertIn("past the end", msg)
        self.assertIn("2 lines", msg)

    def test_inverted_range_is_an_error(self):
        self.write("a.py", "1\n2\n3\n4\n5\n")
        with self.assertRaises(SandboxError):
            tool_read_file(self.cfg, "a.py", start_line=4, end_line=2)

    def test_genuinely_empty_file_still_reports_empty(self):
        self.write("e.txt", "")
        self.assertEqual(tool_read_file(self.cfg, "e.txt"), "(empty file)")

    def test_binary_files_are_refused(self):
        """Decoding a PNG into the conversation is pure context poison."""
        (self.root / "blob.bin").write_bytes(b"\x89PNG\x00\x01\x02binary")
        with self.assertRaises(SandboxError) as ctx:
            tool_read_file(self.cfg, "blob.bin")
        self.assertIn("binary", str(ctx.exception))

    def test_crlf_and_unicode_survive(self):
        self.write("m.txt", "line1\r\nпривет\r\n")
        out = tool_read_file(self.cfg, "m.txt")
        self.assertIn("привет", out)
        self.assertNotIn("\r", out)

    def test_oversized_file_requires_a_range(self):
        self.write("big.txt", "x" * (300 * 1024))
        with self.assertRaises(SandboxError) as ctx:
            tool_read_file(self.cfg, "big.txt")
        self.assertIn("start_line", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
