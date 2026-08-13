"""File tools: editing, searching, and the guards around them."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from lindwyrm.config import Policy
from lindwyrm.sandbox import SandboxError
from lindwyrm.tools import (
    _looks_binary,
    _match_lines,
    tool_edit_file,
    tool_glob,
    tool_grep,
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

    def test_oversized_file_requires_a_range(self):
        self.write("big.txt", "x" * (300 * 1024))
        with self.assertRaises(SandboxError) as ctx:
            tool_read_file(self.cfg, "big.txt")
        self.assertIn("start_line", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
