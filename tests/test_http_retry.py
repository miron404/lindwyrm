"""Retry/backoff behaviour, exercised without touching the network."""

import unittest

from lindwyrm import http as lw_http


class TestBackoffDelay(unittest.TestCase):
    def test_grows_with_each_attempt(self):
        # Jitter makes this probabilistic per-call, so compare the floors.
        first = min(lw_http.backoff_delay(1) for _ in range(50))
        third = min(lw_http.backoff_delay(3) for _ in range(50))
        self.assertGreater(third, first)

    def test_is_capped(self):
        for _ in range(50):
            self.assertLessEqual(lw_http.backoff_delay(20), lw_http.MAX_BACKOFF)

    def test_retry_after_header_wins(self):
        self.assertEqual(lw_http.backoff_delay(1, retry_after=5.0), 5.0)

    def test_retry_after_is_also_capped(self):
        self.assertEqual(
            lw_http.backoff_delay(1, retry_after=9999.0), lw_http.MAX_BACKOFF
        )

    def test_jitter_stays_within_half_of_the_delay(self):
        base = lw_http.DEFAULT_BACKOFF_BASE
        for _ in range(100):
            d = lw_http.backoff_delay(1, base=base)
            self.assertGreaterEqual(d, base * 0.5)
            self.assertLessEqual(d, base)


class TestRetryStatusSet(unittest.TestCase):
    def test_rate_limit_and_server_errors_retry(self):
        for code in (429, 500, 502, 503, 504):
            self.assertIn(code, lw_http.RETRY_STATUS)

    def test_client_errors_do_not_retry(self):
        """A 400 or 401 will fail identically on a retry -- retrying just
        burns time and, for 401, hammers the provider with a bad key."""
        for code in (400, 401, 403, 404, 422):
            self.assertNotIn(code, lw_http.RETRY_STATUS)


class TestClientReuse(unittest.TestCase):
    def tearDown(self):
        lw_http.close_client()

    def test_same_client_is_returned(self):
        self.assertIs(lw_http.get_client(), lw_http.get_client())

    def test_new_client_after_close(self):
        first = lw_http.get_client()
        lw_http.close_client()
        self.assertIsNot(lw_http.get_client(), first)

    def test_close_is_idempotent(self):
        lw_http.get_client()
        lw_http.close_client()
        lw_http.close_client()  # must not raise


if __name__ == "__main__":
    unittest.main()
