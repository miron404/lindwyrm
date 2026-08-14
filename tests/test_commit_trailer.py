"""Git identity and the commit trailer.

Commits go out under someone's name, so the agent must use the identity the
repository already has and never invent one. The trailer is opt-in: marking
work as co-authored by a model is a preference, not a default.
"""

import unittest

from helpers import make_config

from lindwyrm.agent import SYSTEM_PROMPT, Agent, commit_trailer_prompt


class TestTrailerPrompt(unittest.TestCase):
    def test_no_trailer_configured_adds_nothing(self):
        self.assertEqual(commit_trailer_prompt(make_config()), "")

    def test_empty_string_adds_nothing(self):
        self.assertEqual(commit_trailer_prompt(make_config(commit_trailer="  ")), "")

    def test_the_trailer_appears_verbatim(self):
        out = commit_trailer_prompt(
            make_config(commit_trailer="Co-Authored-By: Someone <a@b.c>"))
        self.assertIn("Co-Authored-By: Someone <a@b.c>", out)

    def test_model_is_substituted(self):
        out = commit_trailer_prompt(
            make_config(commit_trailer="by {model}", model="deepseek-v4-pro"))
        self.assertIn("by deepseek-v4-pro", out)
        self.assertNotIn("{model}", out)

    def test_preset_is_substituted(self):
        out = commit_trailer_prompt(
            make_config(commit_trailer="via {preset}", preset_name="flash"))
        self.assertIn("via flash", out)

    def test_it_reaches_the_agent_system_prompt(self):
        agent = Agent(make_config(commit_trailer="Co-Authored-By: X <x@y.z>"))
        self.assertIn("Co-Authored-By: X <x@y.z>", agent.system_prompt)

    def test_an_agent_without_one_is_unchanged(self):
        self.assertEqual(Agent(make_config()).system_prompt, SYSTEM_PROMPT)


class TestIdentityRule(unittest.TestCase):
    """The part that protects the author line, which is always in force."""

    def test_the_prompt_forbids_overriding_the_author(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("--author", lowered)
        self.assertIn("user.name", lowered)

    def test_the_prompt_says_to_ask_rather_than_invent(self):
        self.assertIn("inventing one", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
