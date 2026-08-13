"""Permission resolution: the part that decides what the agent may touch."""

import tempfile
import unittest
from pathlib import Path

from lindwyrm.config import PathRule, Policy, normalize_permission


class TestPolicyDefaults(unittest.TestCase):
    def test_defaults_are_conservative(self):
        p = Policy()
        self.assertEqual(p.read, "allow")
        self.assertEqual(p.write, "confirm")
        self.assertEqual(p.delete, "confirm")
        self.assertEqual(p.bash, "confirm")

    def test_falls_back_to_global_default(self):
        p = Policy()
        self.assertEqual(p.effective("write", Path("/tmp/anything")), "confirm")

    def test_read_only_denies_writes_everywhere(self):
        p = Policy(read_only=True)
        # Even an explicit allow rule must not win over read-only.
        p.set_rule(Path("/tmp"), write="allow", delete="allow")
        self.assertEqual(p.effective("write", Path("/tmp/x")), "deny")
        self.assertEqual(p.effective("delete", Path("/tmp/x")), "deny")
        self.assertEqual(p.effective("read", Path("/tmp/x")), "allow")


class TestRuleSpecificity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "src" / "vendor").mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_most_specific_rule_wins(self):
        p = Policy()
        p.set_rule(self.root / "src", write="allow")
        p.set_rule(self.root / "src" / "vendor", write="deny")
        self.assertEqual(p.effective("write", self.root / "src" / "a.py"), "allow")
        self.assertEqual(
            p.effective("write", self.root / "src" / "vendor" / "a.py"), "deny"
        )

    def test_directory_rule_covers_nested_files(self):
        p = Policy()
        p.set_rule(self.root / "src", write="allow")
        deep = self.root / "src" / "a" / "b" / "c.py"
        self.assertEqual(p.effective("write", deep), "allow")

    def test_unset_operation_inherits_global(self):
        p = Policy()
        p.set_rule(self.root / "src", write="allow")  # delete left as None
        self.assertEqual(p.effective("delete", self.root / "src" / "a.py"), "confirm")

    def test_sibling_directory_is_unaffected(self):
        p = Policy()
        p.set_rule(self.root / "src", write="allow")
        self.assertEqual(p.effective("write", self.root / "other.py"), "confirm")

    def test_set_rule_updates_existing_rule_in_place(self):
        p = Policy()
        p.set_rule(self.root, write="allow")
        p.set_rule(self.root, write="deny")
        self.assertEqual(len(p.rules), 1)
        self.assertEqual(p.effective("write", self.root / "a.py"), "deny")

    def test_clear_rule(self):
        p = Policy()
        p.set_rule(self.root, write="allow")
        self.assertTrue(p.clear_rule(self.root))
        self.assertEqual(p.effective("write", self.root / "a.py"), "confirm")
        self.assertFalse(p.clear_rule(self.root))  # already gone


class TestPermissionNormalization(unittest.TestCase):
    def test_forbid_is_an_alias_for_deny(self):
        self.assertEqual(normalize_permission("forbid"), "deny")
        self.assertEqual(normalize_permission("FORBID"), "deny")

    def test_case_and_whitespace_tolerated(self):
        self.assertEqual(normalize_permission("  Allow "), "allow")


class TestPathRule(unittest.TestCase):
    def test_specificity_grows_with_depth(self):
        shallow = PathRule(path=Path("/a"))
        deep = PathRule(path=Path("/a/b/c"))
        self.assertGreater(deep.specificity(), shallow.specificity())


if __name__ == "__main__":
    unittest.main()
