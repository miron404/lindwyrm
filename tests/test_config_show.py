import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lindwyrm import cli


class TestConfigShow(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.user_config = Path(temp) / "config.toml"
        self.root = (Path(temp) / "project").resolve()
        self.root.mkdir()
        self.project_config = self.root / ".lindwyrm.toml"
        expanduser = os.path.expanduser

        def test_config_path(path):
            if path == "~/.config/lindwyrm/config.toml":
                return str(self.user_config)
            return expanduser(path)

        self.enterContext(mock.patch("lindwyrm.config.os.path.expanduser", test_config_path))
        self.resolve_key = self.enterContext(mock.patch("lindwyrm.config._resolve_preset_api_key"))
        self.make_agent = self.enterContext(mock.patch.object(cli, "_make_agent"))

    def show(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main([*args]), 0)
        self.resolve_key.assert_not_called()
        self.make_agent.assert_not_called()
        return out.getvalue()

    def test_finds_both_paths_and_uses_defaults_when_missing(self):
        out = self.show("config", "show", "-C", str(self.root))
        self.assertIn(f"Global config: {str(self.user_config)!r} (missing)", out)
        self.assertIn(f"Project config: {str(self.project_config)!r} (missing)", out)
        self.assertIn("Project trust: untrusted", out)
        self.assertRegex(out, r"preset + 'deepseek-flash' \[built-in\]")
        self.assertRegex(out, r"proxy + 'direct' \[built-in\]")

    def test_shows_project_overrides_and_refused_values(self):
        self.user_config.write_text('markdown = false\n[policy]\nbash = "confirm"\n')
        self.project_config.write_text(
            'default_preset = "pro"\nproxy = "system"\nthinking = false\n'
            '[policy]\nwrite = "deny"\nbash = "allow"\n')
        out = self.show("-C", str(self.root), "config", "show")
        self.assertRegex(out, r"preset + 'deepseek-pro' \[project default_preset\]")
        self.assertRegex(out, r"thinking + false \[project\]")
        self.assertRegex(out, r"markdown + false \[global\]")
        self.assertRegex(out, r"policy\.write + 'deny' \[project\]")
        self.assertRegex(out, r"policy\.bash + 'confirm' \[global\]")
        self.assertIn("Ignored settings from the untrusted project", out)
        self.assertIn("'proxy'", out)
        self.assertIn("policy.bash =", out)

    def test_reports_cli_and_preset_sources_without_exposing_url_secrets(self):
        self.user_config.write_text(
            'thinking = false\nmax_tokens = 3000\n'
            f'trusted_projects = ["{self.root}"]\n')
        self.project_config.write_text(
            '[[presets]]\nname = "local"\nmodel = "test-model"\n'
            'base_url = "https://user:secret@provider.example/api-secret?token=query-secret"\n'
            'proxy = "http://user:proxy-secret@[::1]:8080"\n')
        out = self.show("-m", "local", "--no-thinking", "--no-save", "--read-only",
                        "--no-markdown", "config", "show", "-C", str(self.root))
        self.assertIn("Project trust: trusted", out)
        self.assertRegex(out, r"preset + 'local' \[--model\]")
        self.assertRegex(out, r"model + 'test-model' \[project preset\]")
        self.assertRegex(out, r"format + 'openai' \[built-in\]")
        self.assertRegex(out, r"max_tokens + 8192 \[built-in\]")
        self.assertRegex(out, r"thinking + false \[--no-thinking\]")
        self.assertRegex(out, r"markdown + false \[--no-markdown\]")
        self.assertRegex(out, r"save_sessions + false \[--no-save\]")
        self.assertRegex(out, r"policy\.read_only + true \[--read-only\]")
        self.assertRegex(out, r"proxy + 'http://\[::1\]:8080' \[project preset\]")
        self.assertIn("'https://provider.example' [project preset]", out)
        self.assertNotIn("secret", out)

    def test_unknown_default_preset_explains_the_legacy_fallback(self):
        self.user_config.write_text('default_preset = "missing"\n')
        out = self.show("--proxy", "system", "config", "show", "-C", str(self.root))
        self.assertRegex(out, r"model + 'missing' \[global default_preset\]")
        self.assertRegex(out, r"proxy + 'system' \[--proxy\]")
        self.assertIn("Unknown default_preset 'missing' [global]", out)
        self.user_config.write_text('model = "legacy-id"\n')
        out = self.show("config", "show", "-C", str(self.root))
        self.assertRegex(out, r"model + 'legacy-id' \[global model\]")
        self.assertNotIn("Unknown default_preset", out)

    def test_trusted_project_preset_list_can_hide_the_global_default(self):
        self.user_config.write_text(
            f'trusted_projects = ["{self.root}"]\ndefault_preset = "global-only"\n'
            '[[presets]]\nname = "global-only"\nmodel = "global-model"\n'
            'base_url = "https://global.example"\n')
        out = self.show("config", "show", "-C", str(self.root))
        self.assertRegex(out, r"preset + 'global-only' \[global default_preset\]")
        self.assertRegex(out, r"model + 'global-model' \[global preset\]")
        self.project_config.write_text("presets = []\n")
        out = self.show("config", "show", "-C", str(self.root))
        self.assertRegex(out, r"preset + 'deepseek-flash' \[built-in\]")
        self.assertIn("Unknown default_preset 'global-only' [global]", out)
