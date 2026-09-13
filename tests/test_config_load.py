"""Settings that come from the config file rather than a preset.

Preset-backed fields are covered in test_preset_fields; these are the plain
top-level keys, where the only thing that can go wrong is that the key is
declared and then never read.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lindwyrm.config import load_config


class ConfigFileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        # A HOME of its own: otherwise the developer's real
        # ~/.config/lindwyrm/config.toml leaks into the test.
        self._home = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {"HOME": self._home.name})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._home.cleanup()
        self._tmp.cleanup()

    def load(self, toml: str = ""):
        if toml:
            (self.root / ".lindwyrm.toml").write_text(toml, encoding="utf-8")
        return load_config(project_root=self.root, require_key=False)


class TestMaxToolSteps(ConfigFileTestCase):
    def test_default(self):
        self.assertEqual(self.load().max_tool_steps, 50)

    def test_read_from_the_project_config(self):
        self.assertEqual(self.load("max_tool_steps = 200\n").max_tool_steps, 200)

    def test_clamped_to_at_least_one(self):
        """Zero would make every turn stop before the first model call."""
        self.assertEqual(self.load("max_tool_steps = 0\n").max_tool_steps, 1)
        self.assertEqual(self.load("max_tool_steps = -5\n").max_tool_steps, 1)


class TestMaxRetries(ConfigFileTestCase):
    def test_default(self):
        self.assertEqual(self.load().max_retries, 4)

    def test_read_and_clamped(self):
        self.assertEqual(self.load("max_retries = 7\n").max_retries, 7)
        self.assertEqual(self.load("max_retries = 0\n").max_retries, 1)


if __name__ == "__main__":
    unittest.main()
