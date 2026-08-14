"""Guards against the bug that bit twice: a new Preset field that some code
path forgets to carry over.

The symptom is nasty because it is silent -- `/model` switches provider and
quietly keeps the previous one's value, with no error anywhere. These tests
fail instead, on the day the field is added.
"""

import dataclasses
import os
import unittest
from unittest import mock

from lindwyrm.config import (
    _PRESET_ONLY_FIELDS,
    Config,
    Preset,
    preset_config_fields,
    preset_overrides,
)


def a_config(**kw) -> Config:
    return Config(api_key="k", **kw)


class TestFieldMapping(unittest.TestCase):
    def test_every_preset_field_is_accounted_for(self):
        """Either it maps onto a Config field, or it is explicitly listed as
        preset-only. There is no third option, and no silent omission."""
        declared = {f.name for f in dataclasses.fields(Preset)}
        covered = set(preset_config_fields()) | set(_PRESET_ONLY_FIELDS)
        self.assertEqual(declared - covered, set())

    def test_mapped_fields_all_exist_on_config(self):
        config_fields = {f.name for f in dataclasses.fields(Config)}
        for name in preset_config_fields():
            self.assertIn(name, config_fields, f"Preset.{name} has no Config field")

    def test_the_list_is_derived_not_written_out(self):
        """If it were hand-written, adding a field to Preset would leave it
        behind -- which is exactly what happened, twice."""
        before = set(preset_config_fields())
        extra = dataclasses.make_dataclass(
            "PresetPlus", [("brand_new_setting", int, dataclasses.field(default=1))],
            bases=(Preset,))
        with mock.patch("lindwyrm.config.Preset", extra):
            after = set(preset_config_fields())
        self.assertIn("brand_new_setting", after - before)


class TestOverrides(unittest.TestCase):
    def setUp(self):
        self.preset = Preset(
            name="other", format="openai", base_url="https://x/v1",
            model="m", thinking=False, max_tokens=4096, thinking_budget=1024,
            temperature=0.3, context_limit=64_000, max_completion_tokens=True,
            price_input=1.0, price_output=2.0, price_cache_read=0.1,
            price_cache_write=1.5, extra_body={"k": "v"},
        )
        self.cfg = a_config(presets={"other": self.preset})

    def test_every_mapped_field_is_transferred(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            values = preset_overrides(self.preset, self.cfg)
        for name in preset_config_fields():
            self.assertEqual(values[name], getattr(self.preset, name), name)

    def test_derived_fields_are_set(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            values = preset_overrides(self.preset, self.cfg)
        self.assertEqual(values["preset_name"], "other")
        self.assertIn("api_key", values)
        self.assertIn("proxy", values)

    def test_extra_body_is_copied_not_shared(self):
        """A preset switched to twice must not accumulate mutations."""
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            values = preset_overrides(self.preset, self.cfg)
        values["extra_body"]["k"] = "changed"
        self.assertEqual(self.preset.extra_body["k"], "v")

    def test_with_preset_applies_the_same_mapping(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            switched = self.cfg.with_preset("other")
        for name in preset_config_fields():
            self.assertEqual(getattr(switched, name), getattr(self.preset, name), name)


class TestInPlaceSwitch(unittest.TestCase):
    """`/model` mutates the shared Config rather than replacing it, and used
    to do so from a separate hand-kept list."""

    def setUp(self):
        self.preset = Preset(name="other", format="openai",
                             base_url="https://x/v1", model="m",
                             context_limit=64_000, max_completion_tokens=True,
                             price_input=9.0, thinking=False)
        self.cfg = a_config(presets={"other": self.preset})

    def switch(self, name):
        from lindwyrm.cli import _switch_preset
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            _switch_preset(self.cfg, name)

    def test_in_place_switch_carries_every_field(self):
        self.switch("other")
        for name in preset_config_fields():
            self.assertEqual(getattr(self.cfg, name), getattr(self.preset, name), name)

    def test_the_fields_that_were_missed_before(self):
        """context_limit and max_completion_tokens were dropped once; the
        prices nearly went the same way."""
        self.switch("other")
        self.assertEqual(self.cfg.context_limit, 64_000)
        self.assertTrue(self.cfg.max_completion_tokens)
        self.assertEqual(self.cfg.price_input, 9.0)

    def test_preset_name_follows(self):
        self.switch("other")
        self.assertEqual(self.cfg.preset_name, "other")

    def test_a_bare_model_id_only_changes_the_model(self):
        """Not a preset name: keep the provider, swap the model string."""
        before = self.cfg.format
        self.switch("some-model-id")
        self.assertEqual(self.cfg.model, "some-model-id")
        self.assertEqual(self.cfg.format, before)


class TestPresetBuilding(unittest.TestCase):
    """_build_presets inherits generically, so a new plain field needs no
    change there either."""

    def build(self, entries):
        from lindwyrm.config import _build_presets
        return _build_presets({"presets": entries})

    def test_new_preset_needs_base_url_and_model(self):
        with self.assertRaises(SystemExit):
            self.build([{"name": "x", "format": "openai"}])

    def test_fields_are_inherited_from_a_builtin_of_the_same_name(self):
        presets = self.build([{"name": "deepseek-flash", "max_tokens": 999}])
        flash = presets["deepseek-flash"]
        self.assertEqual(flash.max_tokens, 999)
        self.assertEqual(flash.base_url, Preset(name="x").base_url)  # inherited

    def test_a_brand_new_preset_starts_with_thinking_off(self):
        """Most OpenAI-compatible endpoints don't support it."""
        presets = self.build([{"name": "n", "base_url": "u", "model": "m"}])
        self.assertFalse(presets["n"].thinking)

    def test_prices_survive_the_generic_path(self):
        presets = self.build([{"name": "n", "base_url": "u", "model": "m",
                               "price_input": 1.25}])
        self.assertEqual(presets["n"].price_input, 1.25)

    def test_a_single_api_key_env_string_becomes_a_tuple(self):
        presets = self.build([{"name": "n", "base_url": "u", "model": "m",
                               "api_key_env": "ONE_VAR"}])
        self.assertEqual(presets["n"].api_key_env, ("ONE_VAR",))

    def test_bad_format_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.build([{"name": "n", "base_url": "u", "model": "m",
                         "format": "graphql"}])

    def test_numeric_fields_are_coerced(self):
        presets = self.build([{"name": "n", "base_url": "u", "model": "m",
                               "max_tokens": "4096", "context_limit": "50000"}])
        self.assertEqual(presets["n"].max_tokens, 4096)
        self.assertEqual(presets["n"].context_limit, 50_000)


if __name__ == "__main__":
    unittest.main()
