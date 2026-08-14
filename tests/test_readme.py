"""The README documents defaults in a table. Tables go stale silently, and a
wrong default in the docs is worse than none -- people tune against it.
"""

import dataclasses
import pathlib
import re
import unittest

from lindwyrm.config import Config

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def documented_defaults() -> dict[str, str]:
    """Parse the `| setting | default | what it does |` table."""
    text = README.read_text(encoding="utf-8")
    found = {}
    for row in re.findall(r"^\| `([a-z_]+)` \| `([^`]+)` \|", text, re.M):
        found[row[0]] = row[1]
    return found


class TestSettingsTable(unittest.TestCase):
    def setUp(self):
        self.documented = documented_defaults()
        self.actual = {f.name: f.default for f in dataclasses.fields(Config)}

    def test_the_table_was_found(self):
        self.assertGreater(len(self.documented), 5,
                           "settings table missing or its shape changed")

    def test_every_documented_setting_exists(self):
        for name in self.documented:
            self.assertIn(name, self.actual, f"README documents no such setting: {name}")

    def test_documented_defaults_match_the_code(self):
        for name, shown in self.documented.items():
            with self.subTest(setting=name):
                real = self.actual[name]
                expected = {"true": True, "false": False}.get(shown.lower())
                if expected is None:
                    expected = type(real)(shown)
                self.assertEqual(real, expected,
                                 f"README says {name} = {shown}, code says {real}")


class TestClaims(unittest.TestCase):
    """A couple of specific numbers the README states in prose."""

    def setUp(self):
        self.text = README.read_text(encoding="utf-8")

    def test_dependency_count_matches_pyproject(self):
        import tomllib
        deps = tomllib.loads(
            (README.parent / "pyproject.toml").read_text())["project"]["dependencies"]
        self.assertEqual(len(deps), 2)
        self.assertIn("two runtime dependencies", self.text)

    def test_every_tool_is_listed(self):
        from lindwyrm.tools import TOOL_SCHEMAS
        for tool in (t["name"] for t in TOOL_SCHEMAS):
            with self.subTest(tool=tool):
                self.assertIn(f"`{tool}`", self.text)

    def test_code_fences_are_balanced(self):
        """A ```bash nested inside a ```markdown block closed it early once,
        and the rest of the section rendered as garbage -- on the PyPI page
        too, since this file is the package description."""
        depth = []
        for line in self.text.splitlines():
            m = re.match(r"^(`{3,})", line)
            if not m:
                continue
            ticks = len(m.group(1))
            if depth and depth[-1] == ticks:
                depth.pop()
            else:
                depth.append(ticks)
        self.assertEqual(depth, [], "unbalanced code fence in README.md")

    def test_contents_links_point_at_real_headings(self):
        headings = {
            re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
            for h in re.findall(r"^##+ (.+)$", self.text, re.M)
        }
        for anchor in re.findall(r"\]\(#([a-z0-9-]+)\)", self.text):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, headings)


if __name__ == "__main__":
    unittest.main()
