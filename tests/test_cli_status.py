"""One-shot mode has to report failure in its exit status.

`lwyrm -p` used to exit 0 after an API error and after work cut off half
way, so a script or CI job carried on as if the agent had succeeded.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import make_config

from lindwyrm import cli, offload
from lindwyrm.agent import Agent
from lindwyrm.client import StreamHandler
from lindwyrm.http import APIError
from lindwyrm.render import Renderer
from lindwyrm.sandbox import UserQuit


def reply(text, stop="end_turn"):
    h = StreamHandler()
    h.content = [{"type": "text", "text": text}]
    h.stop_reason = stop
    h.complete = True
    return h


class TestTurnStatus(unittest.TestCase):
    def run_turn(self, outcome, **cfg):
        agent = Agent(make_config(**cfg))
        if isinstance(outcome, BaseException):
            agent._call_model = mock.Mock(side_effect=outcome)
        else:
            agent._call_model = lambda *a, **kw: outcome
        agent.add_user("go")
        with mock.patch.object(cli, "_persist") as persist, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            status = cli._do_turn(agent.cfg, agent, Renderer(enabled=False))
        return status, persist, out.getvalue()

    def test_success(self):
        status, _, _ = self.run_turn(reply("done"))
        self.assertEqual(status, cli.EXIT_OK)

    def test_api_error(self):
        status, _, _ = self.run_turn(APIError("HTTP 500"))
        self.assertEqual(status, cli.EXIT_ERROR)

    def test_cut_off_reply_is_incomplete_and_says_so(self):
        status, _, out = self.run_turn(reply("1. apple\n2. pea", stop="max_tokens"))
        self.assertEqual(status, cli.EXIT_INCOMPLETE)
        self.assertIn("max_tokens", out)

    def test_interrupt_and_quit(self):
        self.assertEqual(self.run_turn(KeyboardInterrupt())[0], cli.EXIT_INTERRUPTED)
        self.assertEqual(self.run_turn(UserQuit())[0], cli.EXIT_INTERRUPTED)

    def test_the_session_is_saved_on_every_path(self):
        """The history stays valid through an error now, and the tool work
        done before it shouldn't be lost with the terminal."""
        for outcome in (reply("done"), APIError("x"), KeyboardInterrupt()):
            with self.subTest(outcome=type(outcome).__name__):
                _, persist, _ = self.run_turn(outcome)
                persist.assert_called_once()


class TestOneShotExitCode(unittest.TestCase):
    def test_main_returns_the_turn_status(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(project_root=Path(d))
            with mock.patch.object(cli, "load_config", return_value=cfg), \
                    mock.patch.object(cli, "_do_turn", return_value=cli.EXIT_ERROR), \
                    mock.patch.object(offload, "sweep_old_sessions"):
                self.assertEqual(cli.main(["-p", "hi", "--no-save"]), cli.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
