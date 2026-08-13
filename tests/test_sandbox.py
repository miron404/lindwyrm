"""Path resolution: rules must not be dodgeable via `..` or a symlink."""

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from lindwyrm.config import Policy
from lindwyrm.sandbox import SandboxError, authorize, resolve_target


@dataclass
class FakeConfig:
    """Just the two attributes resolve_target/authorize actually read."""
    project_root: Path
    policy: Policy


class TestResolveTarget(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.cfg = FakeConfig(project_root=self.root, policy=Policy())

    def tearDown(self):
        self._tmp.cleanup()

    def test_relative_path_resolves_against_project_root(self):
        self.assertEqual(resolve_target(self.cfg, "a.py"), self.root / "a.py")

    def test_absolute_path_is_kept(self):
        self.assertEqual(resolve_target(self.cfg, "/etc/hosts"), Path("/etc/hosts"))

    def test_dotdot_is_collapsed(self):
        got = resolve_target(self.cfg, "sub/../a.py")
        self.assertEqual(got, self.root / "a.py")

    def test_escaping_dotdot_lands_outside_root(self):
        # Not an error by itself -- but it must resolve, so policy sees the
        # real destination rather than the literal string.
        got = resolve_target(self.cfg, "../outside.py")
        self.assertEqual(got, self.root.parent / "outside.py")

    def test_tilde_is_expanded(self):
        got = resolve_target(self.cfg, "~/x.py")
        self.assertEqual(got, Path(os.path.expanduser("~/x.py")).resolve())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlink_resolves_to_its_target(self):
        secret = self.root / "secret"
        secret.mkdir()
        (self.root / "link").symlink_to(secret)
        got = resolve_target(self.cfg, "link/key.txt")
        self.assertEqual(got, secret / "key.txt")


class TestAuthorize(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_allow_returns_resolved_path_without_prompting(self):
        cfg = FakeConfig(self.root, Policy(read="allow"))
        got = authorize(cfg, "read", "a.py", "read a.py")
        self.assertEqual(got, self.root / "a.py")

    def test_deny_raises(self):
        cfg = FakeConfig(self.root, Policy(write="deny"))
        with self.assertRaises(SandboxError):
            authorize(cfg, "write", "a.py", "write a.py")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unsupported")
    def test_symlink_cannot_dodge_a_deny_rule(self):
        """The whole point of resolving before matching."""
        secret = self.root / "secret"
        secret.mkdir()
        (secret / "key.txt").write_text("shh")
        (self.root / "link").symlink_to(secret)

        policy = Policy(read="allow")
        policy.set_rule(secret, read="deny")
        cfg = FakeConfig(self.root, policy)

        with self.assertRaises(SandboxError):
            authorize(cfg, "read", "secret/key.txt", "direct")
        # Same file reached through the symlink must be denied too.
        with self.assertRaises(SandboxError):
            authorize(cfg, "read", "link/key.txt", "via symlink")

    def test_dotdot_cannot_dodge_a_deny_rule(self):
        outside = self.root.parent / "outside_target"
        policy = Policy(read="allow")
        policy.set_rule(outside, read="deny")
        cfg = FakeConfig(self.root, policy)
        with self.assertRaises(SandboxError):
            authorize(cfg, "read", f"../{outside.name}/x.txt", "via ..")


if __name__ == "__main__":
    unittest.main()
