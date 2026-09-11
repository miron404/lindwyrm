"""Images: reading them, and getting them onto the wire in both formats."""

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from helpers import make_config

from lindwyrm.agent import Agent, tools_for
from lindwyrm.config import Policy
from lindwyrm.openai_client import to_openai_messages
from lindwyrm.sandbox import SandboxError
from lindwyrm.tools import (
    MAX_IMAGE_BYTES,
    ImageResult,
    image_media_type,
    tool_view_image,
)

from dataclasses import dataclass


@dataclass
class FakeConfig:
    project_root: Path
    policy: Policy


def png_bytes(size=8) -> bytes:
    raw = b"".join(b"\x00" + bytes((200, 30, 30)) * size for _ in range(size))
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


class TestMediaTypeSniffing(unittest.TestCase):
    """The provider reads the format from the bytes, not the extension, so a
    .png that is really a JPEG must not be declared as PNG."""

    def test_png(self):
        self.assertEqual(image_media_type(png_bytes()), "image/png")

    def test_jpeg(self):
        self.assertEqual(image_media_type(b"\xff\xd8\xff\xe0rest"), "image/jpeg")

    def test_gif(self):
        self.assertEqual(image_media_type(b"GIF89a...."), "image/gif")

    def test_webp(self):
        self.assertEqual(image_media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")

    def test_not_an_image(self):
        self.assertIsNone(image_media_type(b"#!/usr/bin/env python\n"))

    def test_riff_that_is_not_webp(self):
        self.assertIsNone(image_media_type(b"RIFF\x00\x00\x00\x00WAVEfmt "))


class TestViewImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.cfg = FakeConfig(self.root, Policy(read="allow"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_png_comes_back_as_an_image_result(self):
        (self.root / "shot.png").write_bytes(png_bytes())
        result = tool_view_image(self.cfg, "shot.png")
        self.assertIsInstance(result, ImageResult)
        self.assertEqual(result.media_type, "image/png")
        self.assertIn("shot.png", result.note)

    def test_the_payload_is_base64_of_the_file(self):
        import base64
        data = png_bytes()
        (self.root / "shot.png").write_bytes(data)
        result = tool_view_image(self.cfg, "shot.png")
        self.assertEqual(base64.b64decode(result.data), data)

    def test_a_text_file_is_refused_with_the_reason(self):
        (self.root / "notes.txt").write_text("not a picture")
        with self.assertRaises(SandboxError) as ctx:
            tool_view_image(self.cfg, "notes.txt")
        self.assertIn("not an image", str(ctx.exception))

    def test_a_mislabelled_file_is_judged_by_its_contents(self):
        (self.root / "fake.png").write_bytes(b"\xff\xd8\xffjpeg really")
        self.assertEqual(tool_view_image(self.cfg, "fake.png").media_type, "image/jpeg")

    def test_a_huge_image_is_refused(self):
        """It travels with every later request, so the cap is about request
        size rather than billing."""
        (self.root / "big.png").write_bytes(png_bytes()[:8] + b"\x00" * MAX_IMAGE_BYTES)
        with self.assertRaises(SandboxError) as ctx:
            tool_view_image(self.cfg, "big.png")
        self.assertIn("limit", str(ctx.exception))

    def test_a_missing_file_is_an_error(self):
        with self.assertRaises(SandboxError):
            tool_view_image(self.cfg, "nope.png")

    def test_read_permission_is_enforced(self):
        (self.root / "shot.png").write_bytes(png_bytes())
        self.cfg.policy.read = "deny"
        with self.assertRaises(SandboxError):
            tool_view_image(self.cfg, "shot.png")


class TestToolExposure(unittest.TestCase):
    def test_offered_to_a_model_that_can_see(self):
        names = {t["name"] for t in tools_for(make_config(vision=True))}
        self.assertIn("view_image", names)

    def test_withheld_from_one_that_cannot(self):
        """Otherwise the model spends a turn calling it and apologising."""
        names = {t["name"] for t in tools_for(make_config(vision=False))}
        self.assertNotIn("view_image", names)

    def test_the_other_tools_are_unaffected(self):
        with_vision = {t["name"] for t in tools_for(make_config(vision=True))}
        without = {t["name"] for t in tools_for(make_config(vision=False))}
        self.assertEqual(with_vision - without, {"view_image"})


class TestWireFormats(unittest.TestCase):
    def image_history(self):
        return [
            {"role": "user", "content": [{"type": "text", "text": "look"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "view_image",
                 "input": {"path": "shot.png"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": "QUJD"}},
                    {"type": "text", "text": "[image: shot.png]"}]}]},
        ]

    def test_openai_gets_a_data_url(self):
        out = to_openai_messages("sys", self.image_history())
        tool_msg = [m for m in out if m["role"] == "tool"][0]
        blocks = tool_msg["content"]
        self.assertEqual(blocks[0]["type"], "image_url")
        self.assertEqual(blocks[0]["image_url"]["url"],
                         "data:image/png;base64,QUJD")

    def test_openai_keeps_the_accompanying_note(self):
        out = to_openai_messages("sys", self.image_history())
        blocks = [m for m in out if m["role"] == "tool"][0]["content"]
        self.assertEqual(blocks[1], {"type": "text", "text": "[image: shot.png]"})

    def test_a_text_result_is_still_a_plain_string(self):
        history = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "just text"}]}]
        out = to_openai_messages("sys", history)
        self.assertEqual(out[1]["content"], "just text")


class TestContextAccounting(unittest.TestCase):
    """Base64 of a screenshot is millions of characters but costs the model
    about a thousand tokens. Measuring it literally would trigger compaction
    that frees nothing."""

    def history_with_image(self, payload):
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": payload}}]}]}]

    def test_image_size_barely_moves_the_estimate(self):
        from lindwyrm.agent import estimate_tokens
        small = estimate_tokens(self.history_with_image("A" * 100))
        huge = estimate_tokens(self.history_with_image("A" * 2_000_000))
        self.assertEqual(small, huge)

    def test_an_image_still_counts_for_something(self):
        from lindwyrm.agent import estimate_tokens
        self.assertGreater(estimate_tokens(self.history_with_image("A" * 10)), 500)

    def test_a_huge_image_does_not_trigger_compaction(self):
        agent = Agent(make_config(context_limit=100_000))
        agent.messages = self.history_with_image("A" * 4_000_000) * 4
        self.assertFalse(agent.should_compact())


if __name__ == "__main__":
    unittest.main()
