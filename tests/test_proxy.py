"""Proxy resolution, client pooling, and credential masking.

The rule this pins down: environment proxies are only ever used when asked
for explicitly. A stray ALL_PROXY must never reroute API traffic on its own,
and an explicit "direct" must really be direct.
"""

import os
import unittest
from unittest import mock

from lindwyrm import http as lw_http
from lindwyrm.config import (
    PROXY_DIRECT,
    PROXY_SYSTEM,
    Config,
    Preset,
    mask_proxy,
    parse_proxy,
)


class TestParseProxy(unittest.TestCase):
    def test_urls_pass_through(self):
        for url in ("http://h:8080", "https://h:8080",
                    "socks5://h:1080", "socks5h://h:1080"):
            self.assertEqual(parse_proxy(url, "x"), url)

    def test_direct_spellings(self):
        for raw in ("", "  ", False, None, "direct", "none", "off", "DIRECT"):
            self.assertEqual(parse_proxy(raw, "x"), PROXY_DIRECT)

    def test_system_is_case_insensitive(self):
        self.assertEqual(parse_proxy("System", "x"), PROXY_SYSTEM)

    def test_unknown_scheme_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_proxy("ftp://h:21", "x")

    def test_socks4_says_why(self):
        """httpcore implements socks5 only; the error should say so rather
        than leaving the user to guess."""
        with self.assertRaises(SystemExit) as ctx:
            parse_proxy("socks4://h:1080", "x")
        self.assertIn("socks4", str(ctx.exception))
        self.assertIn("socks5", str(ctx.exception))

    def test_bare_host_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_proxy("127.0.0.1:1080", "x")

    def test_non_string_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_proxy(1234, "x")

    def test_error_names_the_setting(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_proxy("ftp://h", "presets.kimi.proxy")
        self.assertIn("presets.kimi.proxy", str(ctx.exception))


class TestResolution(unittest.TestCase):
    """global x preset x --proxy."""

    def resolve(self, *, global_proxy=PROXY_DIRECT, preset_proxy=None,
                override=None):
        cfg = Config(api_key="k", global_proxy=global_proxy,
                     proxy_override=override)
        return cfg.resolve_proxy(Preset(name="p", proxy=preset_proxy))

    def test_nothing_set_is_direct(self):
        self.assertEqual(self.resolve(), PROXY_DIRECT)

    def test_preset_inherits_global(self):
        self.assertEqual(self.resolve(global_proxy="socks5h://g:1"),
                         "socks5h://g:1")

    def test_preset_overrides_global(self):
        self.assertEqual(
            self.resolve(global_proxy="socks5h://g:1", preset_proxy="http://p:2"),
            "http://p:2")

    def test_preset_can_bypass_the_global_proxy(self):
        """The case an inherit-only model cannot express."""
        self.assertEqual(
            self.resolve(global_proxy="socks5h://g:1", preset_proxy=PROXY_DIRECT),
            PROXY_DIRECT)

    def test_cli_override_beats_the_global(self):
        self.assertEqual(
            self.resolve(global_proxy="socks5h://g:1", override="http://c:3"),
            "http://c:3")

    def test_cli_override_beats_a_preset_proxy(self):
        self.assertEqual(
            self.resolve(preset_proxy="http://p:2", override="http://c:3"),
            "http://c:3")

    def test_cli_override_can_force_direct(self):
        self.assertEqual(
            self.resolve(global_proxy="socks5h://g:1", preset_proxy="http://p:2",
                         override=PROXY_DIRECT),
            PROXY_DIRECT)

    def test_switching_preset_recomputes_the_proxy(self):
        cfg = Config(api_key="k", global_proxy="socks5h://g:1",
                     presets={"a": Preset(name="a", proxy="http://a:1"),
                              "b": Preset(name="b", proxy=PROXY_DIRECT)})
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
            self.assertEqual(cfg.with_preset("a").proxy, "http://a:1")
            self.assertEqual(cfg.with_preset("b").proxy, PROXY_DIRECT)


class TestClientPool(unittest.TestCase):
    def tearDown(self):
        lw_http.close_client()

    def test_same_proxy_reuses_one_client(self):
        self.assertIs(lw_http.get_client("http://h:1"),
                      lw_http.get_client("http://h:1"))

    def test_different_proxies_get_different_clients(self):
        self.assertIsNot(lw_http.get_client("http://h:1"),
                         lw_http.get_client("http://h:2"))

    def test_direct_and_proxied_are_separate(self):
        self.assertIsNot(lw_http.get_client(PROXY_DIRECT),
                         lw_http.get_client("http://h:1"))

    def test_close_clears_every_client(self):
        a = lw_http.get_client(PROXY_DIRECT)
        lw_http.get_client("http://h:1")
        lw_http.close_client()
        self.assertTrue(a.is_closed)
        self.assertIsNot(lw_http.get_client(PROXY_DIRECT), a)

    def test_direct_client_ignores_environment_proxies(self):
        """The whole point of trust_env=False: ALL_PROXY must not silently
        capture API traffic."""
        with mock.patch.dict(os.environ, {"ALL_PROXY": "socks5://127.0.0.1:9"}):
            lw_http.close_client()
            client = lw_http.get_client(PROXY_DIRECT)
            self.assertEqual(len(client._mounts), 0)

    def test_system_client_honors_environment_proxies(self):
        with mock.patch.dict(os.environ, {"ALL_PROXY": "socks5://127.0.0.1:9"}):
            lw_http.close_client()
            client = lw_http.get_client(PROXY_SYSTEM)
            self.assertGreater(len(client._mounts), 0)


class TestLoopbackBypass(unittest.TestCase):
    """A proxy resolves "localhost" on its own side, so proxying a local model
    server would send the request to someone else's machine entirely."""

    def test_loopback_names_and_addresses_always_bypass(self):
        for host in ("localhost", "LOCALHOST", "app.localhost",
                     "127.0.0.1", "127.1.2.3", "::1"):
            self.assertTrue(lw_http.host_bypasses_proxy(host), host)

    def test_public_hosts_do_not_bypass(self):
        for host in ("api.deepseek.com", "8.8.8.8", "notlocalhost.com",
                     "localhost.evil.com"):
            self.assertFalse(lw_http.host_bypasses_proxy(host), host)

    def test_url_for_local_model_goes_direct_despite_a_proxy(self):
        self.assertEqual(
            lw_http.proxy_for_url("http://localhost:11434/v1/chat/completions",
                                  "socks5h://127.0.0.1:1080"),
            PROXY_DIRECT)

    def test_remote_url_still_uses_the_proxy(self):
        self.assertEqual(
            lw_http.proxy_for_url("https://api.deepseek.com/v1/messages",
                                  "socks5h://127.0.0.1:1080"),
            "socks5h://127.0.0.1:1080")

    def test_no_proxy_at_all_stays_direct(self):
        self.assertEqual(
            lw_http.proxy_for_url("https://api.deepseek.com", PROXY_DIRECT),
            PROXY_DIRECT)

    def test_loopback_bypasses_system_proxy_too(self):
        self.assertEqual(
            lw_http.proxy_for_url("http://127.0.0.1:11434", PROXY_SYSTEM),
            PROXY_DIRECT)


class TestNoProxyList(unittest.TestCase):
    def test_exact_host(self):
        self.assertTrue(
            lw_http.host_bypasses_proxy("ollama.box", ["ollama.box"]))

    def test_domain_suffix_matches_subdomains(self):
        for entry in (".internal", "internal"):
            self.assertTrue(
                lw_http.host_bypasses_proxy("gpu.internal", [entry]), entry)
        self.assertFalse(
            lw_http.host_bypasses_proxy("notinternal", [".internal"]))

    def test_cidr_range(self):
        self.assertTrue(
            lw_http.host_bypasses_proxy("192.168.1.50", ["192.168.0.0/16"]))
        self.assertFalse(
            lw_http.host_bypasses_proxy("10.1.1.1", ["192.168.0.0/16"]))

    def test_hostname_never_matches_a_cidr(self):
        self.assertFalse(
            lw_http.host_bypasses_proxy("example.com", ["10.0.0.0/8"]))

    def test_wildcard_bypasses_everything(self):
        self.assertTrue(lw_http.host_bypasses_proxy("api.deepseek.com", ["*"]))

    def test_malformed_entries_are_skipped_not_fatal(self):
        self.assertFalse(
            lw_http.host_bypasses_proxy("api.deepseek.com",
                                        ["", "  ", "999.999/x", "not a cidr/"]))

    def test_lan_model_server_goes_direct(self):
        self.assertEqual(
            lw_http.proxy_for_url("http://192.168.1.50:11434/v1",
                                  "socks5h://127.0.0.1:1080",
                                  ["192.168.0.0/16"]),
            PROXY_DIRECT)


class TestMasking(unittest.TestCase):
    def test_password_is_hidden(self):
        got = mask_proxy("socks5h://bob:hunter2@host:1080")
        self.assertNotIn("hunter2", got)
        self.assertIn("bob", got)
        self.assertIn("host:1080", got)

    def test_url_without_credentials_is_unchanged(self):
        self.assertEqual(mask_proxy("http://host:8080"), "http://host:8080")

    def test_username_only_is_unchanged(self):
        self.assertEqual(mask_proxy("http://bob@host:8080"), "http://bob@host:8080")

    def test_direct_and_system_render_readably(self):
        self.assertEqual(mask_proxy(PROXY_DIRECT), "direct")
        self.assertEqual(mask_proxy(PROXY_SYSTEM), PROXY_SYSTEM)

    def test_redaction_in_error_messages(self):
        proxy = "socks5h://bob:hunter2@host:1080"
        msg = f"ConnectError: failed to connect to {proxy}"
        got = lw_http._redact(msg, proxy)
        self.assertNotIn("hunter2", got)

    def test_redaction_leaves_unrelated_messages_alone(self):
        self.assertEqual(lw_http._redact("boom", "http://h:1"), "boom")


if __name__ == "__main__":
    unittest.main()
