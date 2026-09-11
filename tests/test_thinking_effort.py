"""Reasoning effort.

Providers disagree on how to bound reasoning. Anthropic's API takes a token
budget; DeepSeek accepts that field and ignores it, choosing by effort level
instead -- measured, minimal gave 13k characters of reasoning and max 26k on
the same question.
"""

import unittest

from helpers import make_config

from lindwyrm.client import _build_body as anthropic_body
from lindwyrm.config import THINKING_EFFORTS, parse_effort
from lindwyrm.openai_client import _build_body as openai_body


class TestParsing(unittest.TestCase):
    def test_each_documented_level_is_accepted(self):
        for level in THINKING_EFFORTS:
            self.assertEqual(parse_effort(level), level)

    def test_case_and_space_are_forgiven(self):
        self.assertEqual(parse_effort("  High "), "high")

    def test_unset_stays_unset(self):
        self.assertIsNone(parse_effort(None))
        self.assertIsNone(parse_effort(""))

    def test_a_bad_level_names_the_valid_ones(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_effort("turbo")
        self.assertIn("minimal", str(ctx.exception))


class TestAnthropicBody(unittest.TestCase):
    def body(self, **kw):
        return anthropic_body(make_config(**kw), [], "sys", [])

    def test_effort_is_sent_when_set(self):
        self.assertEqual(self.body(thinking_effort="max")["reasoning_effort"], "max")

    def test_absent_when_unset(self):
        self.assertNotIn("reasoning_effort", self.body())

    def test_not_sent_when_thinking_is_off(self):
        """Nothing to tune when there is no reasoning at all."""
        body = self.body(thinking=False, thinking_effort="max")
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_the_token_budget_is_still_sent(self):
        """Kept for providers that honor it, such as Anthropic's own API."""
        self.assertIn("budget_tokens", self.body(thinking_effort="low")["thinking"])


class TestOpenAIBody(unittest.TestCase):
    def body(self, **kw):
        return openai_body(make_config(**kw), [], [])

    def test_effort_is_sent_when_set(self):
        self.assertEqual(self.body(thinking_effort="low")["reasoning_effort"], "low")

    def test_absent_when_unset(self):
        self.assertNotIn("reasoning_effort", self.body())

    def test_thinking_is_disabled_explicitly(self):
        """The OpenAI path sent no thinking control at all, so a provider
        defaulting to reasoning would reason whatever the config said."""
        self.assertEqual(self.body(thinking=False)["thinking"], {"type": "disabled"})

    def test_nothing_is_sent_when_thinking_is_on(self):
        self.assertNotIn("thinking", self.body(thinking=True))


if __name__ == "__main__":
    unittest.main()
