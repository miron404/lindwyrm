"""Streaming output.

The bug this guards against: rich.Live cannot scroll, so a renderable taller
than the terminal gets cropped with an ellipsis and the display appears to
freeze until the region closes. Everything here is about keeping the live
region bounded and putting finished content into scrollback instead.
"""

import unittest

from lindwyrm.render import _HAS_RICH, TAIL_LINES, Renderer


@unittest.skipUnless(_HAS_RICH, "rich not installed")
class RendererTestCase(unittest.TestCase):
    def setUp(self):
        import io

        from rich.console import Console
        self.renderer = Renderer(enabled=True)
        # A small, fixed terminal makes cropping easy to provoke; the
        # StringIO keeps the rendering out of the test runner's output.
        self.renderer.console = Console(width=60, height=10, record=True,
                                        file=io.StringIO())

    def output(self) -> str:
        return self.renderer.console.export_text()


class TestTail(RendererTestCase):
    def test_tail_is_bounded_by_line_count(self):
        text = "\n".join(f"line {i}" for i in range(500))
        self.assertLessEqual(len(self.renderer._tail(text).splitlines()),
                             TAIL_LINES)

    def test_long_single_line_is_wrapped_and_bounded(self):
        """One 5000-character line is a single logical line but hundreds of
        display rows -- rows are what has to stay under the terminal."""
        tail = self.renderer._tail("x" * 5000)
        rows = tail.splitlines()
        self.assertLessEqual(len(rows), TAIL_LINES)
        for row in rows:
            self.assertLessEqual(len(row), self.renderer.console.width)

    def test_tail_keeps_the_end_not_the_beginning(self):
        text = "\n".join(f"line {i}" for i in range(100))
        self.assertIn("line 99", self.renderer._tail(text))
        self.assertNotIn("line 0\n", self.renderer._tail(text))

    def test_short_text_is_returned_whole(self):
        self.assertEqual(self.renderer._tail("a\nb"), "a\nb")

    def test_blank_lines_are_preserved(self):
        self.assertIn("", self.renderer._tail("a\n\nb").splitlines())


class TestStreaming(RendererTestCase):
    def test_a_long_answer_reaches_scrollback_in_full(self):
        """The whole point: what streamed must end up readable, not cropped."""
        self.renderer.begin_turn()
        for i in range(200):
            self.renderer.on_text(f"sentence number {i}. ")
        self.renderer.end_turn()

        out = self.output()
        self.assertIn("sentence number 0", out)
        self.assertIn("sentence number 199", out)

    def test_answer_is_flushed_before_a_tool_line(self):
        self.renderer.begin_turn()
        self.renderer.on_text("some prose here")
        self.renderer.on_tool("read_file", {"path": "a.py"})
        out = self.output()
        self.assertLess(out.index("some prose"), out.index("read_file"))

    def test_text_buffer_resets_after_flushing(self):
        self.renderer.begin_turn()
        self.renderer.on_text("first part")
        self.renderer.on_tool("bash", {"command": "ls"})
        self.renderer.on_text("second part")
        self.renderer.end_turn()
        # "first part" must appear once, not be re-printed with the second.
        self.assertEqual(self.output().count("first part"), 1)

    def test_no_live_region_is_left_running(self):
        self.renderer.begin_turn()
        self.renderer.on_text("hi")
        self.renderer.end_turn()
        self.assertIsNone(self.renderer._live)


class TestThinkingModes(RendererTestCase):
    def feed(self, mode, chunks=("reasoning step. ",) * 50):
        self.renderer.thinking_mode = mode
        self.renderer.begin_turn()
        for c in chunks:
            self.renderer.on_thinking(c)
        self.renderer.on_text("the answer")
        self.renderer.end_turn()
        return self.output()

    def test_show_keeps_reasoning_in_scrollback(self):
        out = self.feed("show")
        self.assertIn("reasoning step", out)
        self.assertIn("the answer", out)

    def test_peek_leaves_no_reasoning_behind(self):
        out = self.feed("peek")
        self.assertNotIn("reasoning step", out)
        self.assertIn("the answer", out)

    def test_hide_leaves_no_reasoning_behind(self):
        out = self.feed("hide")
        self.assertNotIn("reasoning step", out)
        self.assertIn("the answer", out)

    def test_every_mode_retains_reasoning_for_slash_think(self):
        for mode in ("peek", "show", "hide"):
            with self.subTest(mode=mode):
                self.setUp()
                self.feed(mode)
                self.assertIn("reasoning step", self.renderer.last_thinking)

    def test_hide_still_shows_activity(self):
        """Hiding the content must not look like the process is wedged."""
        self.renderer.thinking_mode = "hide"
        self.renderer.begin_turn()
        self.renderer.on_thinking("secret reasoning")
        self.assertIsNotNone(self.renderer._live)
        self.assertEqual(self.renderer._live_kind, "waiting")


class TestWaitingIndicator(RendererTestCase):
    def test_a_turn_starts_with_an_indicator(self):
        self.renderer.begin_turn()
        self.assertEqual(self.renderer._live_kind, "waiting")

    def test_indicator_gives_way_to_real_output(self):
        self.renderer.begin_turn()
        self.renderer.on_text("hello")
        self.assertEqual(self.renderer._live_kind, "text")

    def test_indicator_returns_after_a_tool_result(self):
        self.renderer.begin_turn()
        self.renderer.on_tool("bash", {"command": "ls"})
        self.renderer.after_tool_result()
        self.assertEqual(self.renderer._live_kind, "waiting")

    def test_waiting_does_not_displace_live_output(self):
        self.renderer.begin_turn()
        self.renderer.on_text("streaming")
        self.renderer.wait()
        self.assertEqual(self.renderer._live_kind, "text")


class TestPlainFallback(unittest.TestCase):
    def test_disabled_renderer_does_not_need_rich(self):
        renderer = Renderer(enabled=False)
        renderer.begin_turn()
        renderer.on_thinking("t")
        renderer.on_text("x")
        renderer.end_turn()
        self.assertIn("t", renderer.last_thinking)


if __name__ == "__main__":
    unittest.main()
